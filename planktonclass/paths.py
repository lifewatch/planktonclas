"""
Miscellaneous functions manage paths.

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification
"""

import os.path
from datetime import datetime

from planktonclass import config

CONF = None
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _get_conf():
    if CONF is not None:
        return CONF
    return config.get_conf_dict()


def get_base_dir():
    conf = _get_conf()
    base_dir = conf["general"]["base_directory"]
    if os.path.isabs(base_dir):
        return base_dir
    else:
        return os.path.abspath(os.path.join(config.CONFIG_ROOT, base_dir))


def get_images_dir():
    conf = _get_conf()
    img_dir = conf["general"]["images_directory"]
    if os.path.isabs(img_dir):
        return img_dir
    else:
        return os.path.abspath(os.path.join(config.CONFIG_ROOT, img_dir))


def get_splits_dir():
    return os.path.join(get_base_dir(), "data", "dataset_files")


def get_models_dir():
    return os.path.join(get_base_dir(), "models")


def get_timestamped_dir():
    return os.path.join(get_models_dir(), timestamp)


def get_checkpoints_dir():
    return os.path.join(get_timestamped_dir(), "ckpts")


def get_logs_dir():
    return os.path.join(get_timestamped_dir(), "logs")


def get_conf_dir():
    return os.path.join(get_timestamped_dir(), "conf")


def get_stats_dir():
    return os.path.join(get_timestamped_dir(), "stats")


def get_ts_splits_dir():
    return os.path.join(get_timestamped_dir(), "dataset_files")


def get_predictions_dir():
    conf = _get_conf()
    file_location = conf.get("testing", {}).get("file_location", None)
    output_directory = conf["testing"]["output_directory"]
    # if file_location is None:
    #     if output_directory is None:
    #         # Define your get_timestamped_dir() function accordingly
    #         return os.path.join(get_timestamped_dir(), "predictions")
    #     else:
    #         return os.path.join(output_directory)
    if file_location is not None:
        if os.path.exists(file_location):
            return os.path.join(os.path.dirname(file_location), "predictions")
    else:
        if output_directory is None:
            # Define your get_timestamped_dir() function accordingly
            return os.path.join(get_timestamped_dir(), "predictions")
        else:
            return os.path.join(output_directory)


def get_results_dir():
    return os.path.join(get_timestamped_dir(), "results")


def get_dirs():
    return {
        "base dir": get_base_dir(),
        "images dir": get_images_dir(),
        "data splits dir": get_splits_dir(),
        "models_dir": get_models_dir(),
        "timestamped dir": get_timestamped_dir(),
        "logs dir": get_logs_dir(),
        "checkpoints dir": get_checkpoints_dir(),
        "configuration dir": get_conf_dir(),
        "statistics dir": get_stats_dir(),
        "timestamped data splits dir": get_ts_splits_dir(),
        "predictions dir": get_predictions_dir(),
        "results dir": get_results_dir(),
    }


def print_dirs():
    dirs = get_dirs()
    max_len = max([len(v) for v in dirs.keys()])
    for k, v in dirs.items():
        print("{k:{l:d}s} {v:3s}".format(l=max_len + 5, v=v, k=k))


def main():
    return print_dirs()


if __name__ == "__main__":
    main()
