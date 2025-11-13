# Isan Pin AI - Configuration

# Model Configuration
MODEL_CONFIG = {
    "base_model": "facebook/musicgen-small",
    "model_name": "isan-pin-musicgen",
    "sample_rate": 32000,
    "channels": 1,
    "duration": 30,  # seconds
    "max_length": 1500,  # tokens
}

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "num_epochs": 50,
    "warmup_steps": 1000,
    "scheduler": "cosine_with_restarts",
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "mixed_precision": True,
    "gradient_clipping": 1.0,
}

# Data Configuration
DATA_CONFIG = {
    "dataset_size": 1000,
    "train_split": 0.8,
    "val_split": 0.15,
    "test_split": 0.05,
    "sample_rate": 48000,
    "bit_depth": 24,
    "min_duration": 30,
    "max_duration": 180,
}

# Audio Processing
AUDIO_CONFIG = {
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 20,
    "fmax": 8000,
    "window": "hann",
}

# Style Definitions
STYLE_DEFINITIONS = {
    "lam_plearn": {
        "thai_name": "หมอลำเพลิน",
        "description": "Slow, contemplative style",
        "tempo_range": [60, 90],
        "mood": ["sad", "romantic", "contemplative"],
    },
    "lam_sing": {
        "thai_name": "หมอลำซิ่ง",
        "description": "Fast, energetic style",
        "tempo_range": [120, 160],
        "mood": ["fun", "exciting", "energetic"],
    },
    "lam_klorn": {
        "thai_name": "หมอลำกลอน",
        "description": "Poetic, storytelling style",
        "tempo_range": [80, 120],
        "mood": ["romantic", "dramatic", "nostalgic"],
    },
    "lam_tad": {
        "thai_name": "ลำตัด",
        "description": "Cutting, abrupt style",
        "tempo_range": [100, 140],
        "mood": ["dramatic", "intense", "powerful"],
    },
    "lam_puen": {
        "thai_name": "ลำพื้น",
        "description": "Traditional, grounded style",
        "tempo_range": [70, 110],
        "mood": ["traditional", "authentic", "earthy"],
    },
}

# Regional Styles
REGIONAL_STYLES = {
    "northeast_north": {
        "thai_name": "อีสานเหนือ",
        "characteristics": ["melodic", "ornamented", "graceful"],
        "instruments": ["pin", "khaen", "pong lang"],
    },
    "northeast_south": {
        "thai_name": "อีสานใต้",
        "characteristics": ["rhythmic", "energetic", "driving"],
        "instruments": ["pin", "khaen", "drums"],
    },
    "northeast_central": {
        "thai_name": "อีสานกลาง",
        "characteristics": ["balanced", "versatile", "expressive"],
        "instruments": ["pin", "khaen", "pong lang", "drums"],
    },
}

# Mood Definitions
MOOD_DEFINITIONS = {
    "sad": {
        "thai_name": "เศร้า",
        "description": "Melancholic, emotional, touching",
        "musical_features": ["minor_key", "slow_tempo", "descending_melodies"],
    },
    "fun": {
        "thai_name": "สนุกสนาน",
        "description": "Joyful, entertaining, lively",
        "musical_features": ["major_key", "fast_tempo", "ascending_melodies"],
    },
    "romantic": {
        "thai_name": "โรแมนติก",
        "description": "Loving, tender, intimate",
        "musical_features": ["soft_dynamics", "legato", "expressive_vibrato"],
    },
    "exciting": {
        "thai_name": "ตื่นเต้น",
        "description": "Thrilling, energetic, intense",
        "musical_features": ["dynamics_changes", "syncopation", "ornamentation"],
    },
    "contemplative": {
        "thai_name": "ครุ่นคิด",
        "description": "Thoughtful, meditative, reflective",
        "musical_features": ["slow_tempo", "sustained_notes", "minimal_ornamentation"],
    },
}

# Technical Parameters
TECHNICAL_CONFIG = {
    "device": "cuda",
    "mixed_precision": True,
    "compile_model": False,
    "torch_compile_mode": "default",
    "seed": 42,
    "deterministic": True,
}

# Evaluation Metrics
EVALUATION_CONFIG = {
    "objective_metrics": ["FAD", "MCD", "SNR", "STFT_loss"],
    "subjective_metrics": ["authenticity", "musicality", "style_accuracy", "satisfaction"],
    "human_evaluation": True,
    "expert_evaluation": True,
    "listening_tests": True,
}

# Web Application
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "allowed_extensions": [".wav", ".mp3", ".flac", ".m4a"],
    "rate_limit": 10,  # requests per minute
}

# API Configuration
API_CONFIG = {
    "version": "v1",
    "title": "Isan Pin AI API",
    "description": "API for generating traditional Isan Pin music using AI",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# Storage Configuration
STORAGE_CONFIG = {
    "audio_dir": "data/audio",
    "models_dir": "models",
    "logs_dir": "logs",
    "temp_dir": "temp",
    "cache_dir": "cache",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Security Configuration
SECURITY_CONFIG = {
    "api_key_required": False,
    "rate_limiting": True,
    "file_validation": True,
    "content_filtering": True,
    "audit_logging": True,
}