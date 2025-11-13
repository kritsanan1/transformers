"""
Utility functions for Isan Pin AI

This module provides:
- Audio processing utilities
- Text processing utilities  
- Helper functions for the main modules
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class AudioUtils:
    """Utility class for audio processing"""
    
    def __init__(self):
        pass
    
    def load_audio(self, file_path: Union[str, Path], sample_rate: int = None, mono: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load audio file with error handling
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate (None for original)
            mono: Whether to convert to mono
            
        Returns:
            Tuple of (audio_signal, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise
    
    def save_audio(self, audio: np.ndarray, file_path: Union[str, Path], sample_rate: int, format: str = "wav"):
        """
        Save audio file
        
        Args:
            audio: Audio signal
            file_path: Output file path
            sample_rate: Sample rate
            format: Audio format
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(file_path, audio, sample_rate, format=format)
            logger.info(f"Saved audio: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save audio {file_path}: {e}")
            raise
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Dict:
        """
        Get audio file information
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            info = sf.info(file_path)
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "subtype": info.subtype,
                "frames": info.frames,
            }
        except Exception as e:
            logger.error(f"Failed to get audio info for {file_path}: {e}")
            return {}
    
    def normalize_audio(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Normalize audio to target RMS level
        
        Args:
            audio: Input audio
            target_rms: Target RMS level
            
        Returns:
            Normalized audio
        """
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            scaling_factor = target_rms / current_rms
            return audio * scaling_factor
        return audio
    
    def trim_silence(self, audio: np.ndarray, sample_rate: int, threshold: float = 0.01, frame_length: int = 2048) -> np.ndarray:
        """
        Trim silence from beginning and end
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            threshold: Silence threshold
            frame_length: Frame length for analysis
            
        Returns:
            Trimmed audio
        """
        try:
            # Find non-silent regions
            intervals = librosa.effects.split(audio, top_db=20, frame_length=frame_length, hop_length=frame_length//4)
            
            if len(intervals) > 0:
                start, end = intervals[0][0], intervals[-1][1]
                return audio[start:end]
            else:
                return audio
                
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio
    
    def apply_fade(self, audio: np.ndarray, sample_rate: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
        """
        Apply fade in/out to audio
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            
        Returns:
            Audio with fade applied
        """
        try:
            audio_faded = audio.copy()
            
            # Apply fade in
            if fade_in > 0:
                fade_in_samples = int(fade_in * sample_rate)
                fade_in_samples = min(fade_in_samples, len(audio_faded))
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                audio_faded[:fade_in_samples] *= fade_in_curve
            
            # Apply fade out
            if fade_out > 0:
                fade_out_samples = int(fade_out * sample_rate)
                fade_out_samples = min(fade_out_samples, len(audio_faded))
                fade_out_curve = np.linspace(1, 0, fade_out_samples)
                audio_faded[-fade_out_samples:] *= fade_out_curve
            
            return audio_faded
            
        except Exception as e:
            logger.warning(f"Fade application failed: {e}")
            return audio
    
    def validate_audio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Validate audio signal
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for NaN or infinite values
            if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                logger.error("Audio contains NaN or infinite values")
                return False
            
            # Check for reasonable amplitude range
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 10.0:
                logger.error(f"Audio amplitude too high: {max_amplitude}")
                return False
            
            # Check minimum length
            min_length = sample_rate // 10  # 0.1 seconds minimum
            if len(audio) < min_length:
                logger.error(f"Audio too short: {len(audio)} samples")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False


class TextProcessor:
    """Utility class for text processing"""
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean text input
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters (basic cleaning)
        import re
        text = re.sub(r'[^\w\sก-๙]', '', text)
        
        return text.strip()
    
    def normalize_description(self, description: str, max_length: int = 512) -> str:
        """
        Normalize description for model input
        
        Args:
            description: Input description
            max_length: Maximum length
            
        Returns:
            Normalized description
        """
        description = self.clean_text(description)
        
        # Truncate if too long
        words = description.split()
        if len(words) > max_length // 4:  # Rough estimate
            description = " ".join(words[:max_length // 4])
        
        return description
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (basic detection)
        
        Args:
            text: Input text
            
        Returns:
            Language code ('th' or 'en')
        """
        if not text:
            return 'en'
        
        # Simple heuristic: check for Thai characters
        import re
        thai_pattern = re.compile(r'[ก-๙]')
        
        if thai_pattern.search(text):
            return 'th'
        else:
            return 'en'
    
    def translate_keywords(self, text: str, target_lang: str = 'en') -> str:
        """
        Translate key music terms (simplified implementation)
        
        Args:
            text: Input text
            target_lang: Target language
            
        Returns:
            Text with translated keywords
        """
        # Basic keyword translation
        thai_to_english = {
            'เสียงพิณ': 'pin sound',
            'หมอลำ': 'molam',
            'เพลิน': 'contemplative',
            'ซิ่ง': 'fast',
            'กลอน': 'poetic',
            'ตัด': 'narrative',
            'ปึน': 'storytelling',
            'เศร้า': 'sad',
            'สนุก': 'fun',
            'โรแมนติก': 'romantic',
            'ช้า': 'slow',
            'เร็ว': 'fast',
            'ปานกลาง': 'medium',
            'ดนตรี': 'music',
            'เพลง': 'song',
            'จังหวะ': 'rhythm',
            'ทำนอง': 'melody',
        }
        
        if target_lang == 'en':
            # Replace Thai terms with English
            for thai, english in thai_to_english.items():
                text = text.replace(thai, english)
        
        return text


def setup_directories(base_dir: str = None) -> Dict[str, Path]:
    """
    Setup necessary directories for the project
    
    Args:
        base_dir: Base directory (uses default if None)
        
    Returns:
        Dictionary of directory paths
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)
    
    directories = {
        'base': base_dir,
        'audio': base_dir / 'audio',
        'models': base_dir / 'models',
        'logs': base_dir / 'logs',
        'temp': base_dir / 'temp',
        'cache': base_dir / 'cache',
        'data': base_dir / 'data',
        'output': base_dir / 'output',
    }
    
    # Create directories
    for dir_path in directories.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Get MD5 hash of a file
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    import hashlib
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human readable string
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


# Make AudioUtils available as a direct import
# This maintains backward compatibility for existing imports
AudioUtils = AudioUtils
TextProcessor = TextProcessor