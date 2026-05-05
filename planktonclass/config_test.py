import os

import yaml

from planktonclass import config

conf_path = config.DEFAULT_CONFIG_PATH
with open(conf_path, "r") as f:
    CONF = yaml.safe_load(f)
