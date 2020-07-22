import os
from dataclasses import dataclass
from hnlp.utils import *


def test_build_class_name():
    assert build_class_name("love-you") == "LoveYou"
    assert build_class_name("love_you") == "LoveYou"
    assert build_class_name("love_You") == "LoveYou"
    assert build_class_name("Love_you") == "LoveYou"
    assert build_class_name("Love-You") == "LoveYou"


def test_check_dir():
    myself = os.path.abspath(__file__)
    try:
        check_dir(myself)
    except Exception as err:
        assert "should be a path" in str(err)
    mydir = os.path.dirname(myself)
    assert check_dir(mydir) == None


def test_check_dir_none():
    try:
        check_dir("")
    except Exception as err:
        assert "should be a path" in str(err)


def test_check_file():
    myself = os.path.abspath(__file__)
    mydir = os.path.dirname(myself)
    try:
        check_file(mydir)
    except Exception as err:
        assert "should be a file" in str(err)
    assert check_file(myself) == None


def test_check_file_none():
    try:
        check_file("")
    except Exception as err:
        assert "should be a file" in str(err)


def test_build_config_from_json():
    json_file = "tests/test_config.json"
    config = build_config_from_json(json_file)
    assert config.name == "Yam"
    assert config.location == "HZ"
    assert config.age == 30


def test_build_pretrained_config_from_json():
    @dataclass
    class Config:
        name: str
        location: str
        age: int
    json_file = "tests/test_config.json"
    config = build_pretrained_config_from_json(Config, json_file)
    assert type(config) == Config
    assert config.name == "Yam"
    assert config.age == 30


def test_get_attr():
    from argparse import Namespace
    typ = Namespace(a=1)
    assert get_attr(typ, "a", 0) == 1
    assert get_attr(typ, "b", 0) == 0

