import torch

ROOT_DIR = "/workdir"


class YoloV5(object):
    def __init__(self, object_id=None):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True, device="cpu"
        )
        self.model.to("cpu")
        if object_id is None:
            object_id = [0, 15]  # person/cat
        self.obj_ids = object_id if isinstance(object_id, list) else [object_id]

    def detect(self, img, threshold=0.1):
        preds = self.model(img)
        cats = [
            f for f in preds.xywh[0] if (f[-1] in self.obj_ids) and (f[-2] > threshold)
        ]
        return len(cats) > 0, preds.render()[0]
