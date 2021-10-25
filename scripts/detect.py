#!/usr/bin/env python3

import torch
import numpy as np
import redis
import time
from datetime import datetime
import os
from PIL import Image

SIZE = (480, 640)
THRESHOLD = 0.5
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


def info(msg):
    print("-" * 100)
    print(msg)
    print("-" * 100)


def run():
    info("Init model")
    rcon = redis.Redis()
    model = YoloV5()
    init_time = time.time()
    rcon.set("num_dets", 0)
    rcon.set("frames", 0)
    rcon.set("duration", 1)
    info("Starting streaming")
    while True:
        data = rcon.get("stream")
        if data is None:
            time.sleep(1)
            continue
        img = np.frombuffer(data, np.uint8).reshape((480, 640, 3))
        cats, render = model.detect(img, threshold=THRESHOLD)
        if cats:
            date = datetime.now()
            folder = os.path.join(f"/workdir/data/{date.strftime('%Y-%m-%d')}")
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"{date.strftime('%Y%m%d%H%M%S%z')}.jpg")
            Image.fromarray(render).save(fname)
            rcon.set("num_dets", int(rcon.get("num_dets")) + 1)
            rcon.set("last_detection", render.tobytes())

        fps = float(rcon.get("frames")) / float(rcon.get("duration"))
        info(f"Currently running at {fps} fps")
        rcon.set("frames", int(rcon.get("frames")) + 1)
        rcon.set("duration", time.time() - init_time)
        rcon.set("render", render.tobytes())


if __name__ == "__main__":
    run()
