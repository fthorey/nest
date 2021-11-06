from box import Box


def load():
    with open("/workdir/config.yaml") as f:
        cfg = Box.from_yaml(f)
    return cfg
