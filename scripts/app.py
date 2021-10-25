import streamlit as st
import numpy as np
import redis
import time

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
live_feed = st.image(get_random_numpy())

st.title("Last detection")
last_det = st.image(get_random_numpy())


def loop():
    rcon = redis.Redis()
    while True:
        if rcon.get("frames") is None:
            time.sleep(1)
            continue
        fps = float(rcon.get("frames")) / float(rcon.get("duration"))
        total = int(rcon.get("frames"))
        txt1.text(f"Total frame analysed: {total}")
        txt2.text(f"Current FPS: {fps}")

        data = rcon.get("render")
        if data is not None:
            render = np.frombuffer(data, np.uint8).reshape((480, 640, 3))
            live_feed.image(render)

        det = rcon.get("last_detection")
        if det is not None:
            det = np.frombuffer(det, np.uint8).reshape((480, 640, 3))
            last_det.image(det)

        time.sleep(1)


if __name__ == "__main__":
    loop()
