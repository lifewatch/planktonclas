"""
Warnings suppression configuration

This module suppresses non-critical warnings from dependencies
to provide a cleaner, user-friendly experience.
"""

import os
import warnings
import logging


APP_LOGGERS = [
    "planktonclass",
    "planktonclass.api",
    "planktonclass.train_runfile",
    "planktonclass.model_utils",
    "planktonclass.data_utils",
    "planktonclass.utils",
]
EPOCH_LOGGER = "planktonclass.epoch_metrics"

class SuppressFilter(logging.Filter):
    """Filter to suppress specific log messages."""
    def filter(self, record):
        # Suppress HDF5 and Keras format warnings from absl
        suppress_messages = [
            'HDF5 file',
            'native Keras format',
            'saving your model',
            'file format is considered legacy'
        ]
        for msg in suppress_messages:
            if msg.lower() in record.getMessage().lower():
                return False
        return True

def configure_warnings():
    """Configure warning filters for cleaner output."""
    
    # Set environment variables before importing TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF INFO and WARNING messages
    os.environ['CUDA_TF_LOG_LEVEL'] = 'OFF'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ.setdefault('TQDM_DISABLE', '0')
    
    # Suppress all warnings globally with simplefilter (most aggressive)
    warnings.simplefilter('ignore')
    
    # Generic warning suppression for common categories
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)
    
    # Specific module suppressions
    warnings.filterwarnings('ignore', module='setuptools.*')
    warnings.filterwarnings('ignore', module='distutils.*')
    warnings.filterwarnings('ignore', module='pkg_resources.*')
    warnings.filterwarnings('ignore', module='pyparsing.*')
    warnings.filterwarnings('ignore', module='marshmallow.*')
    warnings.filterwarnings('ignore', module='apispec.*')
    warnings.filterwarnings('ignore', module='matplotlib.*')
    warnings.filterwarnings('ignore', module='keras.*')
    warnings.filterwarnings('ignore', module='tensorflow.*')
    warnings.filterwarnings('ignore', module='absl.*')
    
    # Specific message suppressions
    warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
    warnings.filterwarnings('ignore', message='.*Implementing implicit namespace packages.*')
    warnings.filterwarnings('ignore', message='.*saving your model as an HDF5 file.*')
    warnings.filterwarnings('ignore', message='.*You are saving your model.*')
    warnings.filterwarnings('ignore', message='.*np.object.*')
    warnings.filterwarnings('ignore', message=".*'missing' attribute.*")
    warnings.filterwarnings('ignore', message='.*You are saving your model.*')
    warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
    warnings.filterwarnings('ignore', message='.*deprecated.*oneOf.*')
    warnings.filterwarnings('ignore', message='.*parseString.*')
    warnings.filterwarnings('ignore', message='.*enablePackrat.*')
    warnings.filterwarnings('ignore', message='.*training configuration.*')
    warnings.filterwarnings('ignore', message='.*No training configuration.*')
    warnings.filterwarnings('ignore', message='.*HDF5 file.*')
    warnings.filterwarnings('ignore', message='.*native Keras format.*')
    
    # Configure logging levels for all known loggers
    for logger_name in [
        'tensorflow',
        'keras',
        'absl',
        'PIL',
        'matplotlib',
        'h5py',
        'flatbuffers',
        'numpy',
        'asyncio',
        'urllib3',
        'urllib3.connectionpool',
        'protobuf'
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        # Remove existing handlers and add a null handler
        logger.handlers = []
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        # Add suppression filter
        logger.addFilter(SuppressFilter())
        # Prevent propagation to root logger
        logger.propagate = False
    
    # Set root logger to ERROR
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    
    # Remove all existing handlers from root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add a null handler to root logger
    root_logger.addHandler(logging.NullHandler())
    
    # Add suppression filter to root logger
    root_logger.addFilter(SuppressFilter())
    
    
    # Configure TensorFlow logging
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as e:
        logging.debug("TensorFlow logging configuration skipped: %s", e)


    
    # Configure application-level logging to show errors and exceptions
    # This needs to happen AFTER library logger suppression
    _configure_app_logging()


def _configure_app_logging():
    """Configure logging for the application itself (not libraries)."""
    # Set up console handler for application loggers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Format with timestamp and level
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Configure planktonclass loggers to show everything
    for logger_name in APP_LOGGERS:
        app_logger = logging.getLogger(logger_name)
        app_logger.setLevel(logging.DEBUG)
        app_logger.propagate = False  # Prevent propagation to root
        # Clear existing handlers
        app_logger.handlers.clear()
        # Add console handler
        app_logger.addHandler(console_handler)

    # Keep epoch metrics out of the console; they are written to file handlers only.
    epoch_logger = logging.getLogger(EPOCH_LOGGER)
    epoch_logger.setLevel(logging.DEBUG)
    epoch_logger.propagate = False
    epoch_logger.handlers.clear()
    epoch_logger.addHandler(logging.NullHandler())


def attach_file_handler(log_path, logger_names=None):
    """Attach a shared file handler to application loggers."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger_names = logger_names or (APP_LOGGERS + [EPOCH_LOGGER])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for logger_name in logger_names:
        app_logger = logging.getLogger(logger_name)
        existing_handler = next(
            (
                handler
                for handler in app_logger.handlers
                if isinstance(handler, logging.FileHandler)
                and getattr(handler, "baseFilename", None)
                == os.path.abspath(log_path)
            ),
            None,
        )
        if existing_handler is not None:
            continue

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)
