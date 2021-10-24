import streamlit as st
from nest import modelzoo
import numpy as np
import redis
import time
from datetime import datetime
import os
from PIL import Image
import threading

SIZE = (480, 640)
THRESHOLD = 0.5
VIEWER_WIDTH = 640
st.title("Who pee on my door step ?")


def get_random_numpy():
    """Return a dummy frame."""
    return np.random.randint(0, 100, size=SIZE)


info = dict()
info["num_dets"] = 0
info["frames"] = 0
info["duration"] = 1

st.title("Live stats")
txt0 = st.text(f"Total detection: 0")
txt1 = st.text(f"Total frame analysed: 0")
txt2 = st.text(f"Current FPS: 0")

st.title("Live feed")
live_feed = st.image(get_random_numpy(), width=VIEWER_WIDTH)

st.title("Last detection")
last_det = st.image(get_random_numpy(), width=VIEWER_WIDTH)


def feed():
    rcon = redis.Redis()
    model = modelzoo.YoloV5()
    init_time = time.time()
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
            info["num_dets"] += 1
            txt0.text(f"Total detection: {info['num_dets']}")

            last_det.image(render, width=VIEWER_WIDTH)

        fps = info["frames"] / info["duration"]
        info["frames"] += 1
        info["duration"] = time.time() - init_time
        txt1.text(f"Total frame analysed: {info['frames']}")
        txt2.text(f"Current FPS: {fps}")
        live_feed.image(render, width=VIEWER_WIDTH)


if __name__ == "__main__":
    feed()
