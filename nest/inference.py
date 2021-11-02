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
