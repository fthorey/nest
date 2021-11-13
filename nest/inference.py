from tritonclient import grpc as triton_client
import tritonclient
import numpy as np
from PIL import Image
from torchvision import transforms as T
from yolov5.utils.general import non_max_suppression
import torch
from yolov5.utils.plots import Annotator, colors
import logging
import time

logger = logging.getLogger(__name__)

NAMES = [
    "person", #0
    "bicycle", #1
    "car", #2
    "motorcycle", #3
    "airplane", #4
    "bus", #5
    "train", #5
    "truck", #7
    "boat", #8
    "traffic light", #9
    "fire hydrant", #10
    "stop sign", #11
    "parking meter", #12
    "bench", #13
    "bird", #14
    "cat", #15
    "dog", #16
    "horse", #17
    "sheep", #18
    "cow", #19
    "elephant", #20
    "bear", #21
    "zebra", #22
    "giraffe", #23
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, x):
        y = np.array(x) / 255.0
        return y.astype(np.float32)


def get_inference_transform(size):
    to_compose = []
    to_compose.append(T.Resize(size))
    to_compose.append(Normalize())
    return T.Compose(to_compose)


def display(img, pred, names=NAMES):
    annotator = Annotator(img, example=str(names))
    for c in pred[:, -1].unique():
        n = (pred[:, -1] == c).sum()  # detections per class
        for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(box, label, color=colors(cls))
    im = annotator.im
    return im.astype(np.uint8)


def wait_for_triton(url, name):
    while True:
        try:
            client = triton_client.InferenceServerClient(url=url)
            client.get_model_config(name)
            break
        except tritonclient.utils.InferenceServerException:
            logger.info("Waiting for the triton server")
            time.sleep(1)


class Inference(object):
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for  persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, name="yolos", url="192.168.0.25:8001"):
        """
        You can find the model on gcs://outflier-outboard/data/comon/triton_models

        Then head to services/outinfer and run `make triton-server`.
        """
        self.name = name
        self.url = url
        self.client = triton_client.InferenceServerClient(url=self.url)
        self.transforms = get_inference_transform(
            self.get_model_config().config.input[0].dims[1:]
        )

    def get_model_config(self):
        return self.client.get_model_config(self.name)

    def _format_image(self, img):
        img = np.expand_dims(self.transforms(img).transpose(2, 0, 1), 0)
        x0 = triton_client.InferInput("input__0", img.shape, "FP32")
        x0.set_data_from_numpy(img)
        return x0

    def _format_request(self, img):
        img = self._format_image(img)
        bbox = triton_client.InferRequestedOutput("output__0")
        return [img], [bbox]

    def _predict(self, img):
        ins, outs = self._format_request(img)
        r = self.client.infer(
            self.name, ins, request_id="0", model_version="0", outputs=outs
        )
        return r.as_numpy("output__0")

    def predict(self, img: np.array):
        img = Image.fromarray(img)
        bbox = self._predict(img)
        bbox = non_max_suppression(
            torch.from_numpy(bbox),
            self.conf,
            iou_thres=self.iou,
            classes=self.classes,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )[0]
        return bbox, display(self.transforms(img) * 255, bbox)
