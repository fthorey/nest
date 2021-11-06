import numpy as np
import redis
import time
from datetime import datetime
import os
from PIL import Image
from nest import inference
import json
import logging

ROOT_DIR = "/workdir"
SLEEP = 1.0 / 30.0

logger = logging.getLogger(__name__)


class Stats:
    def __init__(self):
        self.num_dets = 0
        self.reset()

    def reset(self):
        self.frames = 0
        self.init = time.time()
        self.frame_dets = 0
        self.duration = 0

    def log(self):
        logger.info(f"Average raw FPS: {self.raw_fps}")
        logger.info(f"Average detection FPS: {self.detection_fps}")
        logger.info(f"Detected cat/human: {self.num_dets}")

    def update(self):
        if self.frames % (30 * 5) == 0:
            self.log()
            self.reset()
        self.frame_dets += 1
        self.duration = time.time() - self.init

    @property
    def raw_fps(self):
        return self.frames / float(self.duration) if self.duration != 0 else 0.0

    @property
    def detection_fps(self):
        return self.frame_dets / float(self.duration) if self.duration != 0 else 0.0

    def to_dict(self):
        return {
            "raw_fps": self.raw_fps,
            "detection_fps": self.detection_fps,
            "num_dets": self.num_dets,
        }


class Detectron(object):
    threshold = 0.1

    def __init__(self, object_id=None):
        logger.info("Initializing detectron")
        self.redis_con = redis.Redis()
        self.stats = Stats()
        inference.wait_for_triton("192.168.0.25:8001", "yolovs")
        self.model = inference.Inference(name="yolovs", url="192.168.0.25:8001")
        if object_id is None:
            object_id = [0, 15]  # person/cat
        self.obj_ids = object_id if isinstance(object_id, list) else [object_id]

        sub = self.redis_con.pubsub()
        self._frame = None
        sub.subscribe(**{"video_track": self.handler})
        sub.run_in_thread(sleep_time=SLEEP)
        logger.info("Succesfulluy initialized detectron")

    def handler(self, msg):
        if (msg is None) or (not isinstance(msg["data"], bytes)):
            return None
        self._frame = msg["data"]
        self.stats.frames += 1

    def get_current_frame(self):
        frame = self._frame
        if frame is None:
            return None
        return np.frombuffer(frame, np.uint8).reshape((480, 640, 3))

    def is_worth_saving(self, bbox):
        objs = [
            det
            for det in bbox
            if (det[-1] in self.obj_ids) and (det[-2] > self.threshold)
        ]
        return len(objs) > 0

    def process_one(self, frame):
        bbox, render = self.model.predict(frame)
        self.redis_con.set("detection", render.tobytes())
        if self.is_worth_saving(bbox):
            self.stats.num_dets += 1
            date = datetime.now()
            folder = os.path.join(f"/workdir/data/{date.strftime('%Y-%m-%d')}")
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"{date.strftime('%Y%m%d%H%M%S%z')}.jpg")
            Image.fromarray(render).save(fname)
            self.redis_con.set("last_detection", render.tobytes())

    def stream(self):
        while True:
            frame = self.get_current_frame()
            if frame is None:
                continue
            self.stats.update()
            self.process_one(frame)
            self.redis_con.publish("stats", json.dumps(self.stats.to_dict()))
