"""
Isan Pin AI - AI system for generating traditional Isan Pin music from Northeast Thailand

This package provides a complete pipeline for:
- Data collection and preprocessing of Isan Pin music
- Fine-tuning MusicGen model for Isan Pin music generation
- Generating new Pin music from text descriptions
- Style transfer between different Pin music styles
- Evaluation and quality assessment
- Web and mobile application interfaces

Author: AI Assistant
License: MIT
"""

__version__ = "0.1.0"
__author__ = "AI Assistant"
__email__ = "assistant@example.com"

# Import main modules
from .config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    AUDIO_CONFIG,
    STYLE_DEFINITIONS,
    REGIONAL_STYLES,
    MOOD_DEFINITIONS,
    TECHNICAL_CONFIG,
    EVALUATION_CONFIG,
    WEB_CONFIG,
    API_CONFIG,
    STORAGE_CONFIG,
    LOGGING_CONFIG,
    SECURITY_CONFIG,
)

from .data.collection import DataCollector, AudioClassifier
from .data.preprocessing import AudioPreprocessor, DatasetCreator
from .models.classification import IsanPinClassifier, IsanPinCNN, IsanPinDataset
from .models.musicgen import IsanPinMusicGen
from .inference.inference import IsanPinInference
from .evaluation.evaluator import IsanPinEvaluator
from .web.app import create_gradio_interface, run_web_app
from .utils.audio import AudioUtils

# Main classes for easy import
__all__ = [
    # Configuration
    "MODEL_CONFIG",
    "TRAINING_CONFIG", 
    "DATA_CONFIG",
    "AUDIO_CONFIG",
    "STYLE_DEFINITIONS",
    "REGIONAL_STYLES",
    "MOOD_DEFINITIONS",
    "TECHNICAL_CONFIG",
    "EVALUATION_CONFIG",
    "WEB_CONFIG",
    "API_CONFIG",
    "STORAGE_CONFIG",
    "LOGGING_CONFIG",
    "SECURITY_CONFIG",
    
    # Data modules
    "DataCollector",
    "AudioClassifier",
    "AudioPreprocessor",
    "DatasetCreator",
    
    # Models and training
    "IsanPinClassifier",
    "IsanPinCNN",
    "IsanPinDataset",
    "IsanPinMusicGen",
    
    # Inference
    "IsanPinInference",
    
    # Evaluation
    "IsanPinEvaluator",
    
    # Web app
    "create_gradio_interface",
    "run_web_app",
    
    # Utilities
    "AudioUtils",
]

# Version info
def get_version():
    """Get the current version of Isan Pin AI"""
    return __version__

def get_config_summary():
    """Get a summary of all configuration settings"""
    return {
        "model": MODEL_CONFIG["model_name"],
        "sample_rate": MODEL_CONFIG["sample_rate"],
        "training_batch_size": TRAINING_CONFIG["batch_size"],
        "dataset_size": DATA_CONFIG["dataset_size"],
        "supported_styles": list(STYLE_DEFINITIONS.keys()),
        "supported_regions": list(REGIONAL_STYLES.keys()),
        "supported_moods": list(MOOD_DEFINITIONS.keys()),
    }

# Initialize logging
import logging
import os
from .config import LOGGING_CONFIG, STORAGE_CONFIG

# Create storage directories
os.makedirs(STORAGE_CONFIG["audio_dir"], exist_ok=True)
os.makedirs(STORAGE_CONFIG["models_dir"], exist_ok=True)
os.makedirs(STORAGE_CONFIG["logs_dir"], exist_ok=True)
os.makedirs(STORAGE_CONFIG["temp_dir"], exist_ok=True)
os.makedirs(STORAGE_CONFIG["cache_dir"], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.StreamHandler(),  # Console logging
        logging.FileHandler(
            os.path.join(STORAGE_CONFIG["logs_dir"], "isan_pin_ai.log")
        ) if LOGGING_CONFIG["file_logging"] else logging.NullHandler(),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"Isan Pin AI version {__version__} initialized")
logger.info(f"Configuration summary: {get_config_summary()}")