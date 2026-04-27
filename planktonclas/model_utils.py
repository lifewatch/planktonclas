"""
Miscellanous functions to handle models.

Date: September 2018
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification
"""

import json
import os
import logging
import tarfile
import tempfile

import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.python.saved_model import (
    builder as saved_model_builder, )
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils import (
    predict_signature_def, )

from planktonclas import config, paths, utils

# Configure logger
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME = "Phytoplankton_EfficientNetV2B0"
PRETRAINED_MODEL_TAR = f"{PRETRAINED_MODEL_NAME}.tar.gz"
PRETRAINED_MODEL_URL = (
    f"https://zenodo.org/records/15269453/files/{PRETRAINED_MODEL_TAR}?download=1"
)

model_modes = {
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
    PRETRAINED_MODEL_NAME: "TF",
}


def _safe_extract_tar(archive_path, destination):
    destination = os.path.abspath(destination)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(destination, member.name))
            if os.path.commonpath([destination, member_path]) != destination:
                raise ValueError(f"Unsafe path found in tar archive: {member.name}")
            if member.isdir():
                os.makedirs(member_path, exist_ok=True)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            os.makedirs(os.path.dirname(member_path), exist_ok=True)
            with extracted, open(member_path, "wb") as dst:
                dst.write(extracted.read())


def ensure_pretrained_model(models_dir, modelname=PRETRAINED_MODEL_NAME, force=False):
    """
    Ensure the packaged pretrained model exists locally under the project models directory.
    """
    target_dir = os.path.join(models_dir, modelname)
    os.makedirs(models_dir, exist_ok=True)

    if os.path.isdir(target_dir) and not force:
        return target_dir

    logger.info("Downloading pretrained model: %s", modelname)
    logger.info("Source: %s", PRETRAINED_MODEL_URL)

    temp_archive = None
    try:
        with requests.get(PRETRAINED_MODEL_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                temp_archive = tmp_file.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp_file.write(chunk)

        _safe_extract_tar(temp_archive, models_dir)
    finally:
        if temp_archive and os.path.exists(temp_archive):
            os.remove(temp_archive)

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Pretrained model download completed but expected directory was not created: {target_dir}"
        )

    logger.info("Pretrained model available at: %s", target_dir)
    return target_dir


def create_model(CONF):
    """
    Parameters
    ----------
    CONF : dict
        Contains relevant configuration parameters of the model
    """
    modelname = CONF["model"]["modelname"]

    # Check if a local pre-trained model exists
    models_dir = paths.get_models_dir()
    if modelname == PRETRAINED_MODEL_NAME:
        ensure_pretrained_model(models_dir)
    local_model_path = os.path.join(models_dir, modelname)

    # Try to load from local directory first
    if os.path.isdir(local_model_path):
        logger.info("✓ Found locally trained model: %s", modelname)
        # Look for inference checkpoint in the model directory
        ckpt_dir = os.path.join(local_model_path, "ckpts")

        ckpt_files = sorted(
            [
                f for f in os.listdir(ckpt_dir)
                if f.endswith((".keras", ".h5"))
            ]
        )

        if not ckpt_files:
            raise FileNotFoundError(f"No .keras or .h5 model found in {ckpt_dir}")

        keras_files = [f for f in ckpt_files if f.endswith(".keras")]
        selected_file = keras_files[-1] if keras_files else ckpt_files[-1]
        model_file = os.path.join(ckpt_dir, selected_file)
        logger.info("▌ Loading model checkpoint: %s", os.path.basename(model_file))
        base_model = load_model(model_file, custom_objects=utils.get_custom_objects())
        logger.debug("✓ Model output shape: %s", base_model.output_shape)

        new_input = base_model.input
        x = Dense(CONF["model"]["num_classes"], activation="softmax", name="new_dense")(base_model.layers[-1].output)
        model = Model(inputs=new_input, outputs=x)
        # Add L2 regularization for all the layers in the whole model
        if CONF["training"]["l2_reg"]:
            for layer in model.layers:
                layer.kernel_regularizer = regularizers.l2(
                    CONF["training"]["l2_reg"])
        
        return model, base_model
    
    # Fall back to tf.keras.applications if no local model found
    architecture = getattr(applications, modelname)

    # create the base pre-trained model
    img_width, img_height = (
        CONF["model"]["image_size"],
        CONF["model"]["image_size"],
    )
    base_model = architecture(
        weights="imagenet",
        include_top=False,
        input_shape=(img_width, img_height, 3),
    )

    # Add custom layers at the top to adapt it to our problem
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x) #might work better on large dataset than
    # GlobalAveragePooling https://github.com/keras-team/keras/issues/8470
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(CONF["model"]["num_classes"], activation="softmax")(x)

    # Full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Add L2 reguralization for all the layers in the whole model
    if CONF["training"]["l2_reg"]:
        for layer in model.layers:
            layer.kernel_regularizer = regularizers.l2(
                CONF["training"]["l2_reg"])

    return model, base_model



def save_conf(conf):
    """
    Save CONF to a txt file to ease the reading and to a json file to ease the parsing.

    Parameters
    ----------
    conf : 1-level nested dict
    """
    save_dir = paths.get_conf_dir()

    # Save dict as json file
    with open(os.path.join(save_dir, "conf.json"), "w") as outfile:
        json.dump(conf, outfile, sort_keys=True, indent=4)

    # Save dict as txt file for easier redability
    txt_file = open(os.path.join(save_dir, "conf.txt"), "w")
    txt_file.write("{:<25}{:<30}{:<30} \n".format("group", "key", "value"))
    txt_file.write("=" * 75 + "\n")
    for key, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            txt_file.write("{:<25}{:<30}{:<15} \n".format(
                key, g_key, str(g_val)))
        txt_file.write("-" * 75 + "\n")
    txt_file.close()

