import asyncio
import errno
import logging
import threading
import time
from typing import Optional, Set
import av
from av import VideoFrame
from aiortc.contrib.media import (
    PlayerStreamTrack,
    MediaStreamTrack,
    REAL_TIME_FORMATS,
)
import redis

logger = logging.getLogger(__name__)


def player_worker(
    loop, container, streams, audio_track, video_track, quit_event, throttle_playback
):

    video_first_pts = None
    redis_con = redis.Redis()
    frame_time = None
    start_time = time.time()

    while not quit_event.is_set():
        try:
            frame = next(container.decode(*streams))
        except (av.AVError, StopIteration) as exc:
            if isinstance(exc, av.FFmpegError) and exc.errno == errno.EAGAIN:
                time.sleep(0.01)
                continue
            if video_track:
                asyncio.run_coroutine_threadsafe(video_track._queue.put(None), loop)
            break

        # read up to 1 second ahead
        if throttle_playback:
            elapsed_time = time.time() - start_time
            if frame_time and frame_time > elapsed_time + 1:
                time.sleep(0.1)

        if isinstance(frame, VideoFrame) and video_track:
            if frame.pts is None:  # pragma: no cover
                logger.warning(
                    "MediaPlayer(%s) Skipping video frame with no pts", container.name
                )
                continue

            # video from a webcam doesn't start at pts 0, cancel out offset
            if video_first_pts is None:
                video_first_pts = frame.pts
            frame.pts -= video_first_pts
            try:
                arr = frame.to_ndarray(format="rgb24")
                redis_con.publish("video_track", arr.tobytes())
            except:
                logger.info("Failed to publish on redis")

            frame_time = frame.time
            asyncio.run_coroutine_threadsafe(video_track._queue.put(frame), loop)


class CustomMediaPlayer:
    def __init__(self, file, format=None, options={}):
        self.__container = av.open(file=file, format=format, mode="r", options=options)
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__streams = []
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None
        for stream in self.__container.streams:
            if stream.type == "audio" and not self.__audio:
                self.__audio = PlayerStreamTrack(self, kind="audio")
                self.__streams.append(stream)
            elif stream.type == "video" and not self.__video:
                self.__video = PlayerStreamTrack(self, kind="video")
                self.__streams.append(stream)

        # check whether we need to throttle playback
        container_format = set(self.__container.format.name.split(","))
        self._throttle_playback = not container_format.intersection(REAL_TIME_FORMATS)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker,
                args=(
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__streams,
                    self.__audio,
                    self.__video,
                    self.__thread_quit,
                    self._throttle_playback,
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"MediaPlayer(%s) {msg}", self.__container.name, *args)
