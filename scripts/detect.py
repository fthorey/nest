#!/usr/bin/env python3
import os
import logging
from nest.detector import Detectron

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    detector = Detectron()
    detector.stream()
