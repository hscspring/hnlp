import collections
from typing import Any
from pathlib import Path
import pnlp

import re

name_split_reg = re.compile(r"[-_]")


def build_class_name(name: str):
    return "".join(map(str.capitalize, name_split_reg.split(name)))


def check_dir(path: str):
    p = Path(path)
    if not path or not p.is_dir():
        raise ValueError(f"hnlp: {path} should be a path.")


def check_file(path: str):
    p = Path(path)
    if not path or not p.is_file():
        raise ValueError(f"hnlp: {path} should be a file.")


def build_config_from_json(json_path: str):
    js = pnlp.read_json(json_path)
    # like argparse.Namespace
    Config = collections.namedtuple("Config", js.keys())
    return Config(**js)


def build_pretrained_config_from_json(
        pretrained_config,
        json_path: str):
    js = pnlp.read_json(json_path)
    return pretrained_config(**js)


def get_attr(typ: type, attr: str, default: Any):
    if not hasattr(typ, attr):
        return default
    return getattr(typ, attr)