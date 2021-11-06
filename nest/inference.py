from tritonclient import grpc as triton_client
import numpy as np
from PIL import Image
from torchvision import transforms as T
from yolov5.utils.general import non_max_suppression
import torch
from yolov5.utils.plots import Annotator, colors

NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
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


class Inference(object):
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, name="yolos", url="192.168.0.25:8001"):
        """
        You can find the model on gcs://outflier-outboard/data/comon/triton_models

        Then head to services/outinfer and run `make triton-server`.
        """
        self.name = name
        self.url = url
        self.transforms = get_inference_transform(
            self.get_model_config().config.input[0].dims[1:]
        )

    def get_model_config(self):
        client = triton_client.InferenceServerClient(url=self.url)
        return client.get_model_config(self.name)

    def _format_image(self, img):
        img = np.expand_dims(self.transforms(img).transpose(2, 0, 1), 0)
        x0 = triton_client.InferInput("input__0", img.shape, "FP32")
        x0.set_data_from_numpy(img)
        return x0

    def _format_request(self, img):
        img = self._format_image(img)
        bbox = triton_client.InferRequestedOutput("output__0")
        return [img], [bbox]

    def predict(self, img):
        client = triton_client.InferenceServerClient(url=self.url)
        ins, outs = self._format_request(img)
        r = client.infer(
            self.name, ins, request_id="0", model_version="0", outputs=outs
        )
        return r.as_numpy("output__0")

    def predict_image(self, img: np.array):
        img = Image.fromarray(img)
        bbox = self.predict(img)
        bbox = non_max_suppression(
            torch.from_numpy(bbox),
            self.conf,
            iou_thres=self.iou,
            classes=self.classes,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )[0]
        return display(self.transforms(img) * 255, bbox)
