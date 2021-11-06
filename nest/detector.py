import numpy as np
import redis
import time
from datetime import datetime
import os
from PIL import Image
from nest import inference, config
import json
import logging
from PIL import Image, ImageFont, ImageDraw


def add_msg(img, msg, size=18):
    img = Image.fromarray(img)
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size, encoding="unic"
    )
    d.text((10, 10), msg, font=fnt, fill=(0, 0, 255, 128), fnt=fnt)
    return np.array(img)


ROOT_DIR = "/workdir"
SLEEP = 0.001

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Stats:
    def __init__(self):
        self.num_dets = 0
        self.frames = 0
        self.init = time.time()
        self.frame_dets = 0
        self.duration = 0

    def log(self):
        logger.info(f"Average raw FPS: {self.raw_fps}")
        logger.info(f"Average detection FPS: {self.detection_fps}")
        logger.info(f"Detected cat/human: {self.num_dets}")

    def update(self):
        self.log()
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
    CFG = config.load()

    def __init__(self):
        logger.info("Initializing detectron")
        self.redis_con = redis.Redis(**self.CFG.redis)
        self.stats = Stats()
        inference.wait_for_triton(
            url=self.CFG.inference.url, name=self.CFG.inference.name
        )
        self.model = inference.Inference(
            name=self.CFG.inference.name, url=self.CFG.inference.url
        )
        object_id = self.CFG.inference.object_id
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
        s = self.CFG.size
        return np.frombuffer(frame, np.uint8).reshape((s[1], s[0], 3))

    def is_worth_saving(self, bbox):
        objs = [
            det
            for det in bbox
            if (det[-1] in self.obj_ids) and (det[-2] > self.CFG.inference.threshold)
        ]
        return len(objs) > 0

    def process_one(self, frame):
        bbox, render = self.model.predict(frame)
        render_streamlit = add_msg(
            render, f"Current FPS: {int(self.stats.detection_fps)}"
        )
        self.redis_con.set("detection", render_streamlit.tobytes())
        if self.is_worth_saving(bbox):
            self.stats.num_dets += 1
            date = datetime.now()
            folder = os.path.join(f"/workdir/data/{date.strftime('%Y-%m-%d')}")
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"{date.strftime('%Y%m%d%H%M%S%z')}.jpg")
            Image.fromarray(render).save(fname)
            last = add_msg(render, f"Nb of detections: {int(self.stats.num_dets)}")
            self.redis_con.set("last_detection", last.tobytes())

    def stream(self):
        self.stats.duration = 0
        while True:
            frame = self.get_current_frame()
            if frame is None:
                continue
            self.stats.update()
            self.process_one(frame)
            self.redis_con.publish("stats", json.dumps(self.stats.to_dict()))
