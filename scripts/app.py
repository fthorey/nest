import logging

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from nest.utils import CustomMediaPlayer
from aiortc.contrib.media import MediaPlayer
import redis
import numpy as np

logger = logging.getLogger(__name__)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def app_streaming():
    """Media streamings"""

    def create_player():
        return CustomMediaPlayer(
            "/dev/sensors/camera",
            format="v4l2",
            options={"video_size": "640x480", "framerate": "30"},
        )

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "yolo"]

        def __init__(self) -> None:
            self.type = "noop"
            self.rcon = redis.Redis()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="rgb24")
            if self.type == "noop":
                pass
            elif self.type == "yolo":
                detection = self.rcon.get("detection")
                if detection is not None:
                    img = np.frombuffer(detection, np.uint8).reshape((480, 640, 3))

            return av.VideoFrame.from_ndarray(img, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key=f"media-streaming-hello",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_processor_factory=OpenCVVideoProcessor,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("noop", "yolo")
        )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    app_streaming()
