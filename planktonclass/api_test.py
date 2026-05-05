from planktonclass import config


def get_metadata(distribution_name="planktonclass"):
    """
    Function to read metadata
    """

    metadata = {
        "name": config.MODEL_METADATA.get("name"),
        "author": config.MODEL_METADATA.get("authors"),
        "author-email": config.MODEL_METADATA.get("author-emails"),
        "description": config.MODEL_METADATA.get("summary"),
        "license": config.MODEL_METADATA.get("license"),
        "version": config.MODEL_METADATA.get("version"),
    }

    return metadata
