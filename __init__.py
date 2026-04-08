import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import JinaV4, JinaV4Config 

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the model from HuggingFace.

    Args:
        model_name: the name of the model to download (HuggingFace repo ID)
        model_path: the absolute directory to which to download the model
    """
    logger.info(f"Downloading {model_name} to {model_path}")
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name, model_path, **kwargs):
    """Loads the Jina v4 model.

    Args:
        model_name: the name of the model to load
        model_path: the absolute directory where the model was downloaded
        **kwargs: optional keyword arguments including:
            - classes: list of class names for zero-shot classification (optional)
            - task: "retrieval", "text-matching", or "code" (default: "retrieval")
            - query_prompt_name: prompt name for queries (default: "query")
            - passage_prompt_name: prompt name for passages (default: "passage")
            - pooling_strategy: "mean" or "max" (default: "mean")
            - text_prompt: optional text prompt prefix for classification

    Returns:
        a JinaV4 instance
    """
    # Start with base configuration
    config_dict = {
        "model_path": model_path,
    }
    
    # CRITICAL: Merge all kwargs into config_dict
    config_dict.update(kwargs)
    
    # Create config and model
    config = JinaV4Config(config_dict)
    return JinaV4(config)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    pass