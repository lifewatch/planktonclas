"""
Training runfile

Date: September 2023
Author: Wout Decrop (based on code from Ignacio Heredia)
Email: wout.decrop@VLIZ.be
Github: lifewatch

Description:
This file contains the commands for training a convolutional net for image
classification for phytoplankton.

Additional notes:
* On the training routine: Preliminary tests show that using a custom lr
  multiplier for the lower layers yield to better results than freezing them at
  the beginning and unfreezing them after a few epochs like it is suggested in
  the Keras tutorials.
"""

import io
import json
import logging
import os
import sys
import time
from datetime import datetime
import argparse
import numpy as np

# Configure warnings before importing TensorFlow/Keras.
from planktonclas import warnings_config

warnings_config.configure_warnings()

import tensorflow as tf

from planktonclas import config, model_utils, paths, utils
from planktonclas.data_utils import (
    compute_classweights,
    compute_meanRGB,
    create_data_splits,
    data_sequence,
    json_friendly,
    k_crop_data_sequence,
    load_aphia_ids,
    load_class_names,
    load_data_splits,
)
from planktonclas.optimizers import customAdam

# TODO: Add additional metrics for test time in addition to accuracy

# from planktonclas.api import load_inference_model

# Set TensorFlow verbosity logs
tf.get_logger().setLevel(logging.ERROR)

# Configure logger for training
logger = logging.getLogger("planktonclas.train_runfile")
logger.setLevel(logging.INFO)

# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def log_section(title):
    # line = "=" * 70
    # logger.info(line)
    logger.info("[train] %s", title)
    # logger.info(line)


def log_step(message, *args):
    logger.info("[train] " + message, *args)


def display_path(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    except ValueError:
        return path


def train_fn(TIMESTAMP, CONF):

    paths.timestamp = TIMESTAMP
    paths.CONF = CONF

    utils.create_dir_tree()
    run_log_path = os.path.join(paths.get_logs_dir(), "training.log")
    warnings_config.attach_file_handler(run_log_path)
    utils.backup_splits()
    log_step("Writing run log to: %s", display_path(run_log_path))

    if "train.txt" not in os.listdir(paths.get_ts_splits_dir()):
        if not CONF["dataset"]["split_ratios"]:
            if CONF["training"]["use_validation"] & CONF["training"]["use_test"]:
                split_ratios = [0.8, 0.1, 0.1]
            elif CONF["training"]["use_validation"] & ~CONF["training"]["use_test"]:
                split_ratios = [0.9, 0.1, 0]
            else:
                split_ratios = [1, 0, 0]
        else:
            split_ratios = CONF["dataset"]["split_ratios"]

        log_section("Preparing dataset splits")
        log_step("Split ratios: %s", split_ratios)
        create_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_ratios=split_ratios,
        )
    else:
        log_section("Using existing dataset splits")

    log_section("Loading training data")
    log_step("Splits directory: %s", display_path(paths.get_ts_splits_dir()))
    log_step("Images directory: %s", display_path(paths.get_images_dir()))
    X_train, y_train = load_data_splits(
        splits_dir=paths.get_ts_splits_dir(),
        im_dir=paths.get_images_dir(),
        split_name="train",
    )

    if (
        CONF["training"]["use_validation"]
        and "val.txt" in os.listdir(paths.get_ts_splits_dir())
    ):
        X_val, y_val = load_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_name="val",
        )
    else:
        logger.warning(
            "[train] No validation split found; continuing without validation."
        )
        X_val, y_val = None, None
        CONF["training"]["use_validation"] = False

    class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())
    aphia_ids = load_aphia_ids(splits_dir=paths.get_ts_splits_dir())

    CONF["model"]["preprocess_mode"] = model_utils.model_modes[
        CONF["model"]["modelname"]
    ]
    CONF["training"]["batch_size"] = min(
        CONF["training"]["batch_size"], len(X_train)
    )

    if CONF["model"]["num_classes"] is None:
        CONF["model"]["num_classes"] = len(class_names)

    if CONF["training"]["use_class_weights"]:
        log_section("Computing class weights")
        class_weights = compute_classweights(
            y_train, max_dim=CONF["model"]["num_classes"])
    else:
        class_weights = None

    if CONF["dataset"]["mean_RGB"] is None:
        log_section("Computing dataset statistics")
        CONF["dataset"]["mean_RGB"], CONF["dataset"]["std_RGB"] = compute_meanRGB(
            X_train,
            workers=CONF.get("dataset", {}).get("num_workers", 4)
        )

    train_gen = data_sequence(
        X_train,
        y_train,
        batch_size=CONF["training"]["batch_size"],
        num_classes=CONF["model"]["num_classes"],
        im_size=CONF["model"]["image_size"],
        mean_RGB=CONF["dataset"]["mean_RGB"],
        std_RGB=CONF["dataset"]["std_RGB"],
        preprocess_mode=CONF["model"]["preprocess_mode"],
        aug_params=CONF["augmentation"]["train_mode"],
    )
    train_steps = int(np.ceil(len(X_train) / CONF["training"]["batch_size"]))

    if CONF["training"]["use_validation"]:
        val_gen = data_sequence(
            X_val,
            y_val,
            batch_size=CONF["training"]["batch_size"],
            num_classes=CONF["model"]["num_classes"],
            im_size=CONF["model"]["image_size"],
            mean_RGB=CONF["dataset"]["mean_RGB"],
            std_RGB=CONF["dataset"]["std_RGB"],
            preprocess_mode=CONF["model"]["preprocess_mode"],
            aug_params=CONF["augmentation"]["val_mode"],
        )
        val_steps = int(np.ceil(len(X_val) / CONF["training"]["batch_size"]))
    else:
        val_gen = None
        val_steps = None

    t0 = time.time()

    log_section("Building model")
    model, base_model = model_utils.create_model(CONF)

    base_vars = [var.name for var in base_model.trainable_variables]
    all_vars = [var.name for var in model.trainable_variables]
    top_vars = list(set(all_vars) - set(base_vars))

    if CONF["training"]["mode"] == "fast":
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(
        optimizer=customAdam(
            learning_rate=CONF["training"]["initial_lr"],
            amsgrad=True,
            lr_mult=0.1,
            excluded_vars=top_vars,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log_section("Starting training")
    log_step(
        "Epochs: %s | Batch size: %s | Training samples: %s | Validation samples: %s",
        CONF["training"]["epochs"],
        CONF["training"]["batch_size"],
        len(X_train),
        0 if X_val is None else len(X_val),
    )
    with utils.prefixed_stdout("planktonclas.train_runfile", "[train]"):
        history = model.fit(
            x=train_gen,
            steps_per_epoch=train_steps,
            epochs=CONF["training"]["epochs"],
            class_weight=class_weights,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=utils.get_callbacks(CONF),
            verbose=1,
            initial_epoch=0,
        )

    log_section("Training complete")
    log_step("Saving to: %s", display_path(paths.get_timestamped_dir()))
    log_step("Saving training statistics")
    stats = {
        "epoch": history.epoch,
        "training time (s)": round(time.time() - t0, 2),
        "timestamp": TIMESTAMP,
    }
    stats.update(history.history)
    stats = json_friendly(stats)
    stats_dir = paths.get_stats_dir()
    with open(os.path.join(stats_dir, "stats.json"), "w") as outfile:
        json.dump(stats, outfile, sort_keys=True, indent=4)

    log_step("Saving configuration")
    model_utils.save_conf(CONF)

    log_step("Saving model in HDF5 format")
    fpath = os.path.join(paths.get_checkpoints_dir(), "final_model.h5")

    stderr_backup = sys.stderr
    sys.stderr = io.StringIO()
    try:
        model.save(fpath, include_optimizer=False)
    finally:
        sys.stderr = stderr_backup

    logger.info("[train] Training finished successfully.")

    if CONF["training"]["use_test"]:
        log_section("Evaluating test split")
        X_test, y_test = load_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_name="test",
        )
        crop_num = 10
        filemode = "local"
        test_gen = k_crop_data_sequence(
            inputs=X_test,
            im_size=CONF["model"]["image_size"],
            mean_RGB=CONF["dataset"]["mean_RGB"],
            std_RGB=CONF["dataset"]["std_RGB"],
            preprocess_mode=CONF["model"]["preprocess_mode"],
            aug_params=CONF["augmentation"]["val_mode"],
            crop_mode="random",
            crop_number=crop_num,
            filemode=filemode,
        )
        top_K = 5

        with utils.prefixed_stdout("planktonclas.train_runfile", "[train]"):
            output = model.predict(
                test_gen,
                verbose=1,
                # max_queue_size=10,
                # workers=16,
                # use_multiprocessing=CONF["training"]["use_multiprocessing"],
            )

        output = output.reshape(len(X_test), -1, output.shape[-1])
        output = np.mean(output, axis=1)

        lab = np.argsort(output, axis=1)[:, ::-1]
        lab = lab[:, :top_K]
        prob = output[
            np.repeat(np.arange(len(lab)), lab.shape[1]),
            lab.flatten(),
        ].reshape(lab.shape)

        pred_lab, pred_prob = lab, prob

        if aphia_ids is not None:
            pred_aphia_ids = [aphia_ids[i] for i in pred_lab]
            pred_aphia_ids = [aphia_id.tolist() for aphia_id in pred_aphia_ids]
        else:
            pred_aphia_ids = aphia_ids

        class_index_map = {
            index: class_name for index, class_name in enumerate(class_names)
        }

        pred_lab_names = [
            [class_index_map[label] for label in labels] for labels in pred_lab
        ]
        y_test_names = [class_index_map.get(index) for index in y_test]

        pred_dict = {
            "filenames": list(X_test),
            "pred_lab": pred_lab.tolist(),
            "pred_prob": pred_prob.tolist(),
            "pred_lab_names": pred_lab_names,
            "aphia_ids": pred_aphia_ids,
        }
        if y_test is not None:
            pred_dict["true_lab"] = y_test.tolist()
            pred_dict["true_lab_names"] = y_test_names

        pred_path = os.path.join(
            paths.get_predictions_dir(),
            "{}+{}+top{}.json".format("final_model.h5", "DS_split", top_K),
        )
        with open(pred_path, "w") as outfile:
            json.dump(pred_dict, outfile, sort_keys=True)
        logger.info("[train] Predictions saved to: %s", display_path(pred_path))
        logger.info("[train] Test set evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phytoplankton CNN")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of multiprocessing workers (use 1 for Jupyter)"
    )
    args = parser.parse_args()

    CONF = config.get_conf_dict()
    CONF["dataset"]["num_workers"] = args.workers  # store in CONF
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    train_fn(TIMESTAMP=timestamp, CONF=CONF)