import logging
import os
from PIL import Image

import numpy as np

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.utils.torch import ClassifierOutputProcessor

import torch
import torch.nn.functional as F

from transformers import AutoModel

logger = logging.getLogger(__name__)


class JinaV4Config(fout.TorchImageModelConfig):
    """
    Config class for Jina Embeddings v4.
    
    Jina v4 is a multi-vector retrieval model that can embed both images and text
    in a shared vector space, enabling visual document retrieval and zero-shot classification.
    
    Args:
        model_path (str): HuggingFace model ID. Default: "jinaai/jina-embeddings-v4"
        
        task (str): Task type for embeddings. Options: "retrieval", "text-matching", "code"
            Default: "retrieval"
        
        query_prompt_name (str): Prompt name for queries. Default: "query"
        
        passage_prompt_name (str): Prompt name for passages/documents. Default: "passage"
        
        text_prompt (str): Optional baseline text prompt for classification. Default: ""
        
        pooling_strategy (str): Final pooling strategy for multi-vector to 1D conversion.
            Options: "mean" (default) or "max"
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        # Jina handles preprocessing internally, so use raw inputs
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        
        # Only set up output processor if classes provided (for classification)
        if "classes" in d and d["classes"] is not None and len(d["classes"]) > 0:
            if "output_processor_cls" not in d:
                d["output_processor_cls"] = "fiftyone.utils.torch.ClassifierOutputProcessor"
        
        super().__init__(d)
        
        # Jina-specific configuration
        self.model_path = self.parse_string(d, "model_path", default="jinaai/jina-embeddings-v4")
        self.task = self.parse_string(d, "task", default="retrieval")
        self.query_prompt_name = self.parse_string(d, "query_prompt_name", default="query")
        self.passage_prompt_name = self.parse_string(d, "passage_prompt_name", default="passage")
        self.text_prompt = self.parse_string(d, "text_prompt", default="")
        self.pooling_strategy = self.parse_string(d, "pooling_strategy", default="mean")
        
        # Validate task
        valid_tasks = ["retrieval", "text-matching", "code"]
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{self.task}'")
        
        # Validate pooling strategy
        if self.pooling_strategy not in ["mean", "max"]:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'max', got '{self.pooling_strategy}'"
            )


class JinaV4(fout.TorchImageModel, fom.PromptMixin):
    """
    Jina Embeddings v4 model for document understanding and retrieval.
    
    This model can:
    1. Embed images into vectors (single-vector: 2048 dim, multi-vector: Nx128)
    2. Embed text queries into vectors
    3. Support zero-shot classification via multi-vector similarity
    4. Enable visual document retrieval
    
    It extends TorchImageModel for image processing and PromptMixin for text embedding.
    """
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A JinaV4Config instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Storage for cached data
        self._text_features = None  # Cached text features for classification
        self._last_computed_embeddings = None  # Last computed 1D embeddings
        self._last_computed_multi_vector_embeddings = None  # Multi-vector embeddings (list)
        
        self.pooling_strategy = config.pooling_strategy

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings."""
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts."""
        return True
    
    @property
    def classes(self):
        """The list of class labels for the model."""
        return self._classes

    @classes.setter
    def classes(self, value):
        """Set new classes and invalidate cached text features."""
        self._classes = value
        self._text_features = None  # Invalidate cache
        
        # Rebuild output processor if classes are provided
        if value is not None and len(value) > 0:
            self._output_processor = ClassifierOutputProcessor(classes=value)
        else:
            self._output_processor = None
    
    @property
    def text_prompt(self):
        """The text prompt prefix for classification."""
        return self.config.text_prompt

    @text_prompt.setter  
    def text_prompt(self, value):
        """Set new text prompt and invalidate cached text features."""
        self.config.text_prompt = value
        self._text_features = None  # Invalidate cache
    
    def _apply_final_pooling(self, embeddings):
        """Apply final pooling to multi-vector embeddings.
        
        Reduces multi-vector embeddings to single vectors for FiftyOne storage.
        
        Args:
            embeddings: Multi-vector embeddings, either:
                - torch.Tensor of shape (batch, num_vectors, dim)
                - List of tensors with shapes [(num_vectors, dim), ...]
            
        Returns:
            torch.Tensor: Fixed-dimension pooled embeddings with shape (batch, dim)
        """
        # Handle list of variable-length tensors
        if isinstance(embeddings, list):
            pooled_list = []
            for emb in embeddings:
                # emb shape: (num_vectors, dim)
                if self.pooling_strategy == "mean":
                    pooled = emb.mean(dim=0)  # (dim,)
                elif self.pooling_strategy == "max":
                    pooled = emb.max(dim=0)[0]  # (dim,)
                else:
                    raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")
                pooled_list.append(pooled)
            return torch.stack(pooled_list)  # (batch, dim)
        
        # Handle batched tensor
        if self.pooling_strategy == "mean":
            pooled = embeddings.mean(dim=1)  # (batch, dim)
            return pooled
        elif self.pooling_strategy == "max":
            pooled = embeddings.max(dim=1)[0]  # (batch, dim)
            return pooled
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

    def _load_model(self, config):
        """Load Jina v4 model from HuggingFace.
        
        Args:
            config: JinaV4Config instance containing model parameters

        Returns:
            The loaded model ready for inference
        """
        
        
        logger.info(f"Loading Jina v4 model from {config.model_path}")
        
        # Determine dtype based on device
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load model
        model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        
        model.to(self._device)
        model.eval()
        
        logger.info(f"Model loaded on {self._device} with dtype {torch_dtype}")
        
        return model

    def _prepare_images_for_jina(self, imgs):
        """Convert images to PIL format (Jina's expected input).
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            List of PIL Images
        """
        pil_images = []
        
        for img in imgs:
            if isinstance(img, Image.Image):
                # Already PIL Image
                pil_images.append(img)
            elif isinstance(img, torch.Tensor):
                # Tensor (CHW) → PIL Image
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = img_np.astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                # Numpy array (HWC) → PIL Image
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        return pil_images

    def _get_text_features(self):
        """Get or compute text features for classification.
        
        Creates embeddings for each class by combining text_prompt with class names.
        
        Returns:
            List of multi-vector text embeddings (one per class)
        """
        if self._text_features is None:
            # Create prompts for each class
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            # Compute and cache multi-vector text features
            self._text_features = self._embed_prompts(prompts)
        
        return self._text_features
    
    def _embed_prompts(self, prompts):
        """Embed text prompts using Jina's encode_text() with multi-vectors.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            List of multi-vector embeddings [(num_vectors, 128), ...]
        """
        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts=prompts,
                task=self.config.task,
                prompt_name=self.config.query_prompt_name,
                return_multivector=True
            )
        
        # embeddings is a list of tensors on CUDA
        # Each tensor has shape (num_vectors, 128)
        return embeddings

    def embed_prompt(self, prompt):
        """Embed a single text prompt to 1D vector for retrieval.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: 1D embedding vector
        """
        with torch.no_grad():
            # Get single-vector embedding (2048 dim)
            embeddings = self.model.encode_text(
                texts=[prompt],
                task=self.config.task,
                prompt_name=self.config.query_prompt_name,
                return_multivector=False
            )
        
        # embeddings is [tensor(2048,)] on CUDA
        result = embeddings[0].cpu().numpy()
        return result

    def embed_prompts(self, prompts):
        """Embed multiple text prompts to 1D vectors for retrieval.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: 1D embeddings with shape (batch, 2048)
        """
        with torch.no_grad():
            # Get single-vector embeddings
            embeddings = self.model.encode_text(
                texts=prompts,
                task=self.config.task,
                prompt_name=self.config.query_prompt_name,
                return_multivector=False
            )
        
        # Stack and convert to numpy
        result = torch.stack(embeddings).cpu().numpy()
        return result

    def embed_images(self, imgs):
        """Embed a batch of images.
        
        Returns 1D embeddings for retrieval, and stores multi-vector embeddings
        internally for classification.
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            numpy array: 1D embeddings with shape (batch, dim)
        """
        # Convert to PIL Images
        pil_images = self._prepare_images_for_jina(imgs)
        
        with torch.no_grad():
            # Get multi-vector embeddings for classification
            multivector_embeddings = self.model.encode_image(
                images=pil_images,
                task=self.config.task,
                return_multivector=True
            )
            
            # Store multi-vector embeddings (list of variable-length tensors)
            self._last_computed_multi_vector_embeddings = multivector_embeddings
            
            # Apply final pooling to get 1D vectors for retrieval
            final_embeddings = self._apply_final_pooling(multivector_embeddings)
            
            # Cache final embeddings
            self._last_computed_embeddings = final_embeddings
        
        # Return as numpy array
        result = final_embeddings.cpu().numpy()
        return result
    
    def embed(self, img):
        """Embed a single image.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: 1D embedding
        """
        embeddings = self.embed_images([img])
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed
            
        Returns:
            numpy array: 1D embeddings
        """
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed 1D embeddings.
        
        Returns:
            numpy array: The last computed 1D embeddings
        """
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
        
        result = self._last_computed_embeddings.cpu().numpy()
        return result

    def _get_class_logits(self, text_features, image_features):
        """Calculate multi-vector similarity scores using MaxSim.
        
        Implements ColBERT-style MaxSim scoring for multi-vector embeddings.
        
        Args:
            text_features: List of multi-vector text embeddings
                [(num_text_vectors, 128), ...] - one per class
            image_features: List of multi-vector image embeddings
                [(num_image_vectors, 128), ...] - one per image
            
        Returns:
            tuple: (logits_per_image, logits_per_text)
                - logits_per_image: shape (num_images, num_classes)
                - logits_per_text: shape (num_classes, num_images)
        """
        with torch.no_grad():
            num_classes = len(text_features)
            num_images = len(image_features)
            
            # Compute scores for each (class, image) pair
            scores = torch.zeros(num_classes, num_images, device=self._device)
            
            for i, text_emb in enumerate(text_features):
                # text_emb: (num_text_vectors, 128)
                for j, image_emb in enumerate(image_features):
                    # image_emb: (num_image_vectors, 128)
                    
                    # Compute pairwise similarities
                    # (num_text_vectors, 128) @ (128, num_image_vectors)
                    similarities = torch.matmul(text_emb, image_emb.t())
                    # similarities: (num_text_vectors, num_image_vectors)
                    
                    # MaxSim: for each text vector, find max similarity to any image vector
                    max_sims = similarities.max(dim=1)[0]  # (num_text_vectors,)
                    
                    # Sum across text vectors
                    score = max_sims.sum()
                    scores[i, j] = score
            
            logits_per_text = scores  # (num_classes, num_images)
            logits_per_image = scores.t()  # (num_images, num_classes)
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run zero-shot classification on a batch of images.
        
        Uses multi-vector similarity between image and class text embeddings.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            Classification predictions
        """
        # Check if classification is supported
        if self.classes is None or len(self.classes) == 0:
            raise ValueError(
                "Cannot perform classification without classes. "
                "Set classes: model.classes = ['class1', 'class2', ...]"
            )
        
        if self._output_processor is None:
            raise ValueError(
                "No output processor configured for classification."
            )
        
        # Get image embeddings (stores multi-vector embeddings internally)
        _ = self.embed_images(imgs)
        
        # Get multi-vector embeddings
        image_features = self._last_computed_multi_vector_embeddings
        text_features = self._get_text_features()
        
        # Calculate multi-vector similarity
        output, _ = self._get_class_logits(text_features, image_features)
        
        # Get frame size for output processor
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].size()[-2:]
        elif hasattr(imgs[0], 'size'):  # PIL Image
            width, height = imgs[0].size
        else:
            height, width = imgs[0].shape[:2]  # numpy array
        
        frame_size = (width, height)
        
        if self.has_logits:
            self._output_processor.store_logits = self.store_logits
        
        return self._output_processor(
            output, 
            frame_size, 
            confidence_thresh=self.config.confidence_thresh
        )