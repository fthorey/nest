#!/usr/bin/env python3
import os
import logging
from nest import deployer, config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.INFO)
    cfg = config.load()
    deployer.to_triton(size=(cfg.size[1], cfg.size[0]))
