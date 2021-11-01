import logging

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av
import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from tritonclient import grpc as triton_client
import numpy as np
from PIL import Image


class Inference(object):
    def __init__(self, name="big-lama-best", url="0.0.0.0:8001"):
        """
        This model rely on the triton server with the big-lama-best model
        to be up a and running.

        You can find the model on gcs://outflier-outboard/data/comon/triton_models/big-lama-best

        Then head to services/outinfer and run `make triton-server`.
        """
        self.name = name
        self.url = url

    @staticmethod
    def _format_image(img):
        img = np.array(Image.fromarray(img).convert(mode="RGB"))
        img = img.transpose(2, 0, 1).astype("float32") / 255.0
        batch = img[np.newaxis, :, :, :]
        x0 = triton_client.InferInput("input__0", batch.shape, "FP32")
        x0.set_data_from_numpy(batch)
        return x0

    @staticmethod
    def _format_mask(mask):
        mask = (
            np.array(Image.fromarray(mask).convert(mode="L")).astype("float32") / 255.0
        )
        batch = mask[np.newaxis, np.newaxis, :, :]
        x1 = triton_client.InferInput("input__1", batch.shape, "FP32")
        x1.set_data_from_numpy(batch)

        return x1

    def _format_request(self, img, mask):
        x0 = self._format_image(img)
        x1 = self._format_mask(mask)
        out = triton_client.InferRequestedOutput("output__0")
        return [x0, x1], [out]

    def predict(self, img: np.array, mask: np.array):
        """Inpaint an image according to a mask using LAMA

        :params img: img to be inpainted
        :params mask: mask to guide the inpainting
        :returns: the inpainted image
        """
        client = triton_client.InferenceServerClient(url=self.url)
        ins, outs = self._format_request(img, mask.squeeze() == 1.0)
        r = client.infer(
            self.name, ins, request_id="0", model_version="0", outputs=outs
        )
        return (r.as_numpy("output__0") * 255).astype("uint8")[0].transpose(1, 2, 0)


logger = logging.getLogger(__name__)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def app_streaming():
    """Media streamings"""

    def create_player():
        return MediaPlayer(
            "/dev/sensors/camera",
            format="v4l2",
            options={"video_size": "640x480", "framerate": "30"},
        )

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "yolo"]

        def __init__(self) -> None:
            self.type = "noop"
            self.model = Inference()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="rgb24")
            if self.type == "noop":
                pass
            elif self.type == "yolo" and self.model is not None:
                mask = np.zeros(img.shape[:-1])
                h, w = mask.shape
                mask[h // 2 - 100 : h // 2 + 100, w // 2 - 100 : w // 2 + 100] = 1
                mask = cv2.resize(mask, (640, 360))
                img = cv2.resize(img, (640, 360))
                img = self.model.predict(img, mask)
                img = cv2.resize(img, (640, 480))
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
