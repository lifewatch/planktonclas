"""
Configuration script of the image classification application. It load configuration from a YAML file.

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification
"""

import builtins
import os
import ntpath
import posixpath
import re
import textwrap

import yaml

from importlib import metadata

MODEL_NAME = os.getenv("MODEL_NAME", default="planktonclas")
MODEL_METADATA = metadata.metadata(MODEL_NAME)
# Fix metadata for authors from pyproject parsing
_EMAILS = MODEL_METADATA["Author-email"].split(", ")
_EMAILS = map(lambda s: s[:-1].split(" <"), _EMAILS)
MODEL_METADATA["Author-emails"] = dict(_EMAILS)

# Fix metadata for authors from pyproject parsing
_AUTHORS = MODEL_METADATA.get("Author", "").split(", ")
_AUTHORS = [] if _AUTHORS == [""] else _AUTHORS
_AUTHORS += MODEL_METADATA["Author-emails"].keys()
MODEL_METADATA["Authors"] = sorted(_AUTHORS)

homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(homedir, "planktonclas", "resources", "config.yaml")
CONFIG_ENV_VAR = "PLANKTONCLAS_CONFIG"
CONF_PATH = None
CONFIG_ROOT = homedir
CONF = None

MODEL_PREPROCESS_MODES = {
    "DenseNet121": "torch",
    "DenseNet169": "torch",
    "DenseNet201": "torch",
    "InceptionResNetV2": "tf",
    "EfficientNetV2B0": "tf",
    "InceptionV3": "tf",
    "MobileNet": "tf",
    "NASNetLarge": "tf",
    "NASNetMobile": "tf",
    "Xception": "tf",
    "ResNet50": "caffe",
    "VGG16": "caffe",
    "VGG19": "caffe",
    "Phytoplankton_EfficientNetV2B0": "tf",
    "FlowCam": "tf",
    "FlowCyto": "tf",
    "PI10": "tf",
}


def normalize_user_path(path, path_separator=None):
    """
    Normalize user-provided path separators so either slash style is accepted.
    """
    if path is None or not isinstance(path, str):
        return path

    separator = os.sep if path_separator is None else path_separator
    normalized = path.replace("\\", separator).replace("/", separator)
    path_module = ntpath if separator == "\\" else posixpath
    return path_module.normpath(normalized)


def apply_internal_defaults(conf_d):
    """
    Apply runtime-only defaults that should not be exposed in the user config.
    """
    model_conf = conf_d.setdefault("model", {})
    pretrained_conf = conf_d.setdefault("pretrained", {})
    training_conf = conf_d.setdefault("training", {})
    general_conf = conf_d.setdefault("general", {})
    testing_conf = conf_d.setdefault("testing", {})

    modelname = model_conf.get("modelname")
    if "preprocess_mode" not in model_conf:
        if modelname is not None:
            model_conf["preprocess_mode"] = MODEL_PREPROCESS_MODES.get(modelname, "tf")
        else:
            model_conf["preprocess_mode"] = "tf"

    for key in ("base_directory", "images_directory"):
        if key in general_conf:
            general_conf[key] = normalize_user_path(general_conf[key])

    for key in ("output_directory", "file_location"):
        if key in testing_conf:
            testing_conf[key] = normalize_user_path(testing_conf[key])

    model_conf.setdefault("num_classes", None)
    pretrained_conf.setdefault("use_pretrained", False)
    pretrained_conf.setdefault("name", None)
    pretrained_conf.setdefault("version", "latest")
    training_conf.setdefault("lr_schedule_mode", "step")
    return conf_d


def check_conf(conf=None):
    """
    Checks for configuration parameters
    """
    conf = CONF if conf is None else conf
    for group, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            gg_keys = g_val.keys()

            if g_val["value"] is None:
                continue

            if "type" in gg_keys:
                var_type = getattr(builtins, g_val["type"])
                if not isinstance(g_val["value"], var_type):
                    raise TypeError(
                        "The selected value for {} must be a {}.".format(
                            g_key, g_val["type"]))

            if ("choices" in gg_keys) and (g_val["value"]
                                           not in g_val["choices"]):
                raise ValueError(
                    "The selected value for {} is not an available choice.".
                    format(g_key))

            if "range" in gg_keys:
                if (g_val["range"][0] is not None) and (g_val["range"][0]
                                                        > g_val["value"]):
                    raise ValueError(
                        "The selected value for {} is lower than the minimal possible value."
                        .format(g_key))

                if (g_val["range"][1] != "None") and (g_val["range"][1]
                                                      < g_val["value"]):
                    raise ValueError(
                        "The selected value for {} is higher than the maximal possible value."
                        .format(g_key))

    # Check augmentation dict
    for d_name in ["train_mode", "val_mode"]:
        d = conf["augmentation"][d_name]["value"]

        if (d is None) or (not d):
            continue

        for k in [
                "h_flip",
                "v_flip",
                "stretch",
                "crop",
                "zoom",
                "blur",
                "pixel_noise",
                "pixel_sat",
                "cutout",
                "rot",
        ]:
            if not isinstance(d[k], float):
                raise TypeError(
                    "The type of the {} key in the {} augmentation dict must be float."
                    .format(k, d_name))

            if not (0 <= d[k] <= 1):
                raise TypeError(
                    "The {} key in the {} augmentation dict must be in the [0, 1] range."
                    .format(k, d_name))

        if not isinstance(d["rot_lim"], int):
            raise TypeError(
                "The {} key in the {} augmentation dict must be an int.".
                format("rot_lim", d_name))


_DOUBLE_QUOTED_BACKSLASH_PATTERN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_VALID_YAML_ESCAPES = set("0abtnvfreN_LPuxU\"\\/")


def _escape_invalid_yaml_backslashes(raw_text):
    """
    Preserve Windows-style backslashes inside double-quoted YAML scalars.
    """

    def _normalize_scalar(match):
        scalar = match.group(1)
        chunks = []
        index = 0
        while index < len(scalar):
            char = scalar[index]
            if char != "\\":
                chunks.append(char)
                index += 1
                continue

            next_index = index + 1
            if next_index >= len(scalar):
                chunks.append("\\\\")
                index += 1
                continue

            next_char = scalar[next_index]
            if next_char in _VALID_YAML_ESCAPES:
                chunks.append("\\")
                chunks.append(next_char)
            else:
                chunks.append("\\\\")
                chunks.append(next_char)
            index += 2

        return f'"{"".join(chunks)}"'

    return _DOUBLE_QUOTED_BACKSLASH_PATTERN.sub(_normalize_scalar, raw_text)


def load_yaml_config(stream_or_text):
    """
    Load YAML while tolerating Windows backslashes in double-quoted strings.
    """
    raw_text = stream_or_text.read() if hasattr(stream_or_text, "read") else stream_or_text
    try:
        return yaml.safe_load(raw_text)
    except yaml.scanner.ScannerError as exc:
        if "found unknown escape character" not in str(exc):
            raise
        return yaml.safe_load(_escape_invalid_yaml_backslashes(raw_text))


def _load_conf_file(conf_path):
    with open(conf_path, "r", encoding="utf-8") as f:
        return load_yaml_config(f)


def _get_resolution_root(conf_path):
    if os.path.abspath(conf_path) == os.path.abspath(DEFAULT_CONFIG_PATH):
        return os.getcwd()
    return os.path.dirname(os.path.abspath(conf_path))


def set_config_path(conf_path=None):
    """
    Load configuration from disk and refresh the exported module state.
    """
    global CONF_PATH, CONFIG_ROOT, CONF, conf_dict

    selected_path = conf_path or os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    selected_path = os.path.abspath(selected_path)
    CONF_PATH = selected_path
    CONFIG_ROOT = _get_resolution_root(selected_path)
    CONF = _load_conf_file(selected_path)
    check_conf(conf=CONF)
    conf_dict = get_conf_dict(conf=CONF)
    return CONF


def get_conf_dict(conf=None):
    """
    Return configuration as dict
    """
    conf = CONF if conf is None else conf
    conf_d = {}
    for group, val in conf.items():
        conf_d[group] = {}
        for g_key, g_val in val.items():
            conf_d[group][g_key] = g_val["value"]
        # Make dict empty if it's not needed
    if not conf_d["augmentation"]["use_augmentation"]:
        conf_d["augmentation"]["train_mode"] = None
        conf_d["augmentation"]["val_mode"] = None
    return apply_internal_defaults(conf_d)


set_config_path()


def print_full_conf(conf=None):
    """
    Print all configuration parameters (including help, range, choices, ...)
    """
    conf = CONF if conf is None else conf
    for group, val in sorted(conf.items()):
        print("=" * 75)
        print("{}".format(group))
        print("=" * 75)
        for g_key, g_val in sorted(val.items()):
            print("{}".format(g_key))
            for gg_key, gg_val in g_val.items():
                print("{}{}".format(" " * 4, gg_key))
                body = "\n".join([
                    "\n".join(
                        textwrap.wrap(
                            line,
                            width=110,
                            break_long_words=False,
                            replace_whitespace=False,
                            initial_indent=" " * 8,
                            subsequent_indent=" " * 8,
                        )) for line in str(gg_val).splitlines()
                    if line.strip() != ""
                ])
                print(body)
            print("\n")


def print_conf_table(conf=None):
    """
    Print configuration parameters in a table
    """
    conf = conf_dict if conf is None else conf
    print("{:<25}{:<30}{:<30}".format("group", "key", "value"))
    print("=" * 75)
    for group, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            print("{:<25}{:<30}{:<15} \n".format(group, g_key, str(g_val)))
        print("-" * 75 + "\n")
