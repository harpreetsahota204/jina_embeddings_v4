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
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)


class JinaV4Config(fout.TorchImageModelConfig):
    """
    Config class for Jina Embeddings v4.
    
    Jina v4 is a vision-language model that can embed both images and text
    in a shared vector space, enabling visual document retrieval and zero-shot classification.
    
    The model operates in two modes:
    - Single-vector mode: 2048-dim embeddings for retrieval/similarity search
    - Multi-vector mode: Variable Nx128-dim embeddings for fine-grained classification
    
    Args:
        model_path (str): HuggingFace model ID. Default: "jinaai/jina-embeddings-v4"
        
        task (str): Task type for embeddings. Options: "retrieval", "text-matching", "code"
            Default: "retrieval"
        
        query_prompt_name (str): Prompt name for queries. Default: "query"
        
        passage_prompt_name (str): Prompt name for passages/documents. Default: "passage"
        
        text_prompt (str): Optional baseline text prompt for classification. Default: ""
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

        # Validate task
        valid_tasks = ["retrieval", "text-matching", "code"]
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{self.task}'")


class JinaV4(fout.TorchImageModel, fom.PromptMixin):
    """
    Jina Embeddings v4 model for document understanding and retrieval.
    
    This model supports two workflows:
    
    1. **Retrieval/Similarity Search**: Returns 2048-dim single-vector embeddings
       - Use with compute_embeddings() and compute_similarity()
       - Efficient cosine similarity search
       
    2. **Zero-Shot Classification**: Uses variable-length multi-vector embeddings
       - Use with apply_model()
       - MaxSim scoring for fine-grained classification
       - Separate forward pass from retrieval
    
    The model extends TorchImageModel for image processing and PromptMixin for text embedding.
    """
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A JinaV4Config instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Storage for cached data
        self._text_features = None  # Cached multi-vector text features for classification
        self._last_computed_embeddings = None  # Last computed 2048-dim embeddings
        

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
    
    def _load_model(self, config):
        """Load Jina v4 model from HuggingFace.
        
        Args:
            config: JinaV4Config instance containing model parameters

        Returns:
            The loaded model ready for inference
        """
        logger.info(f"Loading Jina v4 model from {config.model_path}")
        
        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self._device)
            
            # Use bfloat16 for Ampere or newer GPUs (capability >= 8.0)
            if capability[0] >= 8:
                model_kwargs["dtype"] = torch.bfloat16
            else:
                model_kwargs["dtype"] = torch.float16
        else:
            # For CPU and MPS (Mac), use float16 as BFloat16 is not supported
            model_kwargs["dtype"] = torch.float16

        # Enable flash attention if available (only on CUDA)
        if is_flash_attn_2_available() and self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = AutoModel.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        
        self.model.to(self._device)
        self.model.eval()
        
        return self.model

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
                    # Assume normalized [0, 1] or [-1, 1]
                    if img_np.min() < 0:
                        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                # Numpy array (HWC) → PIL Image
                if img.dtype != np.uint8:
                    # Assume normalized [0, 1] or [0, 255]
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        return pil_images

    def _get_text_features(self):
        """Get or compute multi-vector text features for classification.
        
        Creates embeddings for each class by combining text_prompt with class names.
        
        Returns:
            List of multi-vector text embeddings [(num_vectors, 128), ...] - one per class
        """
        if self._text_features is None:
            # Create prompts for each class
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            # Compute and cache multi-vector text features
            self._text_features = self._embed_prompts_multivector(prompts)
        
        return self._text_features
    
    def _embed_prompts_multivector(self, prompts):
        """Embed text prompts using multi-vector mode for classification.
        
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
                return_multivector=True  # Multi-vector for classification
            )
        
        # embeddings is a list of tensors on CUDA
        # Each tensor has shape (num_vectors, 128)
        return embeddings

    def embed_prompt(self, prompt):
        """Embed a single text prompt to 2048-dim vector for retrieval.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: 2048-dim embedding vector
        """
        with torch.no_grad():
            # Get single-vector embedding (2048 dim)
            embeddings = self.model.encode_text(
                texts=[prompt],
                task=self.config.task,
                prompt_name=self.config.query_prompt_name,
                return_multivector=False  # Single-vector for retrieval
            )
        
        # embeddings is [tensor(2048,)] on CUDA
        result = embeddings[0].cpu().numpy()
        return result

    def embed_prompts(self, prompts):
        """Embed multiple text prompts to 2048-dim vectors for retrieval.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: 2048-dim embeddings with shape (batch, 2048)
        """
        with torch.no_grad():
            # Get single-vector embeddings
            embeddings = self.model.encode_text(
                texts=prompts,
                task=self.config.task,
                prompt_name=self.config.query_prompt_name,
                return_multivector=False  # Single-vector for retrieval
            )
        
        # Stack and convert to numpy
        result = torch.stack(embeddings).cpu().numpy()
        return result

    def embed_images(self, imgs):
        """Embed images to 2048-dim vectors for retrieval/similarity search.
        
        Uses single-vector mode which returns fixed 2048-dim embeddings.
        These embeddings are normalized and suitable for cosine similarity.
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            numpy array: 2048-dim embeddings with shape (batch, 2048)
        """
        pil_images = self._prepare_images_for_jina(imgs)
        
        with torch.no_grad():
            # Get single-vector embeddings (2048-dim) for retrieval
            embeddings = self.model.encode_image(
                images=pil_images,
                task=self.config.task,
                return_multivector=False  # Single-vector mode
            )
            
            # Stack into tensor
            final_embeddings = torch.stack(embeddings)
            
            # Cache for get_embeddings()
            self._last_computed_embeddings = final_embeddings
        
        return final_embeddings.cpu().numpy()
    
    def embed(self, img):
        """Embed a single image.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: 2048-dim embedding
        """
        embeddings = self.embed_images([img])
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed
            
        Returns:
            numpy array: 2048-dim embeddings
        """
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed 2048-dim embeddings.
        
        Returns:
            numpy array: The last computed embeddings with shape (batch, 2048)
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
        For each text vector, finds the maximum similarity with any image vector,
        then sums across all text vectors.
        
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
                    
                    # Normalize embeddings
                    text_norm = F.normalize(text_emb, p=2, dim=1)
                    image_norm = F.normalize(image_emb, p=2, dim=1)
                    
                    # Compute pairwise similarities
                    # (num_text_vectors, 128) @ (128, num_image_vectors)
                    similarities = torch.matmul(text_norm, image_norm.t())
                    # similarities: (num_text_vectors, num_image_vectors)
                    
                    # MaxSim: for each text vector, find max similarity to any image vector
                    max_sims = similarities.max(dim=1)[0]  # (num_text_vectors,)
                    
                    # Sum across text vectors (mean also works)
                    score = max_sims.sum()
                    scores[i, j] = score
            
            logits_per_text = scores  # (num_classes, num_images)
            logits_per_image = scores.t()  # (num_images, num_classes)
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run zero-shot classification on a batch of images.
        
        Uses multi-vector similarity between image and class text embeddings.
        This performs a separate forward pass from embed_images() to get
        multi-vector representations for fine-grained classification.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            Classification predictions processed by output processor
        """
        # Check if classification is supported
        if self.classes is None or len(self.classes) == 0:
            raise ValueError(
                "Cannot perform classification without classes. "
                "Set classes when loading: foz.load_zoo_model(..., classes=['class1', 'class2'])"
            )
        
        if self._output_processor is None:
            raise ValueError(
                "No output processor configured for classification."
            )
        
        # Convert images to PIL
        pil_images = self._prepare_images_for_jina(imgs)
        
        # Get multi-vector image embeddings (separate forward pass)
        with torch.no_grad():
            image_features = self.model.encode_image(
                images=pil_images,
                task=self.config.task,
                return_multivector=True  # Multi-vector for classification
            )
        
        # Get cached multi-vector text features for classes
        text_features = self._get_text_features()
        
        # Calculate multi-vector similarity using MaxSim
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