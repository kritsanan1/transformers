"""
Data Collection Module for Isan Pin AI

This module handles:
- Audio file collection and organization
- Metadata extraction and management
- Audio format conversion and standardization
- Quality assessment and filtering
"""

import os
import json
import logging
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import torchaudio
from datetime import datetime

from ..config import DATA_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG, LOGGING_CONFIG
from ..utils.audio import AudioUtils
from ..utils.text import TextProcessor

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Collects and preprocesses audio data for Isan Pin AI training
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sample_rate: int = None,
        classifier: Optional['AudioClassifier'] = None
    ):
        """
        Initialize the data collector
        
        Args:
            input_dir: Directory containing raw audio files
            output_dir: Directory for processed output
            sample_rate: Target sample rate (uses config default if None)
            classifier: Optional AudioClassifier for automatic classification
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate or DATA_CONFIG["sample_rate"]
        self.classifier = classifier
        
        # Create output directories
        self.audio_output_dir = self.output_dir / "audio"
        self.metadata_dir = self.output_dir / "metadata"
        self.temp_dir = self.output_dir / "temp"
        
        for dir_path in [self.audio_output_dir, self.metadata_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.audio_utils = AudioUtils()
        self.text_processor = TextProcessor()
        
        logger.info(f"Initialized DataCollector: input={input_dir}, output={output_dir}")
    
    def process_all(self) -> Dict:
        """
        Process all audio files in the input directory
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting batch processing of audio files")
        
        # Find all audio files
        audio_files = self._find_audio_files()
        logger.info(f"Found {len(audio_files)} audio files")
        
        if not audio_files:
            logger.warning("No audio files found in input directory")
            return {"processed": 0, "errors": 0, "total": 0}
        
        # Process files
        results = []
        errors = []
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
            
            try:
                result = self._process_single_file(audio_file)
                results.append(result)
                logger.info(f"Successfully processed: {audio_file.name}")
            except Exception as e:
                errors.append({"file": str(audio_file), "error": str(e)})
                logger.error(f"Error processing {audio_file.name}: {e}")
        
        # Save processing log
        self._save_processing_log(results, errors)
        
        stats = {
            "processed": len(results),
            "errors": len(errors),
            "total": len(audio_files),
            "success_rate": len(results) / len(audio_files) * 100,
        }
        
        logger.info(f"Processing completed: {stats}")
        return stats
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the input directory"""
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.input_dir.rglob(f"*{ext}"))
            audio_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        
        return sorted(list(set(audio_files)))  # Remove duplicates
    
    def _process_single_file(self, audio_file: Path) -> Dict:
        """
        Process a single audio file
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Dictionary with processing results
        """
        logger.debug(f"Processing file: {audio_file}")
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise Exception(f"Failed to load audio: {e}")
        
        # Get audio info
        duration = len(audio) / self.sample_rate
        
        # Validate audio
        if duration < DATA_CONFIG["min_duration"]:
            raise Exception(f"Audio too short: {duration:.2f}s < {DATA_CONFIG['min_duration']}s")
        
        if duration > DATA_CONFIG["max_duration"]:
            logger.warning(f"Audio longer than max: {duration:.2f}s")
            # Optionally trim or split
            audio = audio[:int(DATA_CONFIG["max_duration"] * self.sample_rate)]
            duration = DATA_CONFIG["max_duration"]
        
        # Quality assessment
        quality_metrics = self._assess_audio_quality(audio)
        
        # Generate filename
        output_filename = self._generate_output_filename(audio_file)
        output_path = self.audio_output_dir / output_filename
        
        # Save processed audio
        sf.write(output_path, audio, self.sample_rate)
        
        # Extract or generate metadata
        metadata = self._extract_metadata(audio_file, audio, duration, quality_metrics)
        
        # Classify if classifier is available
        if self.classifier:
            classification = self.classifier.classify(audio)
            metadata.update(classification)
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{output_filename.stem}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        result = {
            "input_file": str(audio_file),
            "output_file": str(output_path),
            "metadata_file": str(metadata_path),
            "duration": duration,
            "sample_rate": self.sample_rate,
            "quality_score": quality_metrics.get("overall_score", 0),
            "processing_time": datetime.now().isoformat(),
        }
        
        return result
    
    def _assess_audio_quality(self, audio: np.ndarray) -> Dict:
        """
        Assess the quality of audio
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        try:
            # Signal-to-noise ratio (simplified)
            signal_power = np.mean(audio ** 2)
            if signal_power > 0:
                # Estimate noise floor (last 10% of signal)
                noise_samples = int(len(audio) * 0.1)
                noise_power = np.mean(audio[-noise_samples:] ** 2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                metrics["snr"] = float(snr)
            else:
                metrics["snr"] = -float('inf')
            
            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            metrics["dynamic_range"] = float(dynamic_range)
            
            # Zero crossing rate (for detecting silence)
            zcr = np.mean(np.abs(np.diff(np.signbit(audio))))
            metrics["zero_crossing_rate"] = float(zcr)
            
            # Overall quality score (0-10)
            # This is a simplified version - in practice, you'd want more sophisticated metrics
            quality_components = []
            
            # SNR component
            if metrics["snr"] > 20:
                quality_components.append(1.0)
            elif metrics["snr"] > 10:
                quality_components.append(0.7)
            else:
                quality_components.append(0.3)
            
            # Dynamic range component
            if dynamic_range > 0.1:
                quality_components.append(1.0)
            elif dynamic_range > 0.05:
                quality_components.append(0.7)
            else:
                quality_components.append(0.3)
            
            # Zero crossing rate component (lower is generally better for music)
            if zcr < 0.3:
                quality_components.append(1.0)
            elif zcr < 0.5:
                quality_components.append(0.7)
            else:
                quality_components.append(0.3)
            
            metrics["overall_score"] = float(np.mean(quality_components) * 10)
            
        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            metrics["overall_score"] = 0.0
        
        return metrics
    
    def _extract_metadata(self, audio_file: Path, audio: np.ndarray, duration: float, quality_metrics: Dict) -> Dict:
        """
        Extract metadata from audio file and signal
        
        Args:
            audio_file: Original audio file path
            audio: Audio signal
            duration: Duration in seconds
            quality_metrics: Quality assessment results
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "filename": audio_file.name,
            "original_path": str(audio_file),
            "duration": duration,
            "sample_rate": self.sample_rate,
            "channels": 1,  # Always mono after processing
            "bit_depth": DATA_CONFIG["bit_depth"],
            "file_size": audio_file.stat().st_size,
            "processing_date": datetime.now().isoformat(),
            "quality_metrics": quality_metrics,
        }
        
        # Extract features using librosa
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            metadata["tempo_bpm"] = float(tempo)
            
            # Key estimation (simplified)
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            key_idx = np.argmax(np.mean(chroma, axis=1))
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            metadata["estimated_key"] = keys[key_idx]
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            metadata["spectral_centroid"] = float(np.mean(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            metadata["spectral_rolloff"] = float(np.mean(spectral_rolloff))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            metadata["zero_crossing_rate"] = float(np.mean(zero_crossing_rate))
            
            # MFCC features (first 13 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            metadata["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
            metadata["mfcc_std"] = np.std(mfcc, axis=1).tolist()
            
        except Exception as e:
            logger.warning(f"Could not extract audio features: {e}")
        
        # Add file metadata
        stat = audio_file.stat()
        metadata["file_metadata"] = {
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
        }
        
        return metadata
    
    def _generate_output_filename(self, input_file: Path) -> str:
        """Generate output filename based on input file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = input_file.stem.replace(" ", "_").replace("-", "_")
        return f"{base_name}_{timestamp}.wav"
    
    def _save_processing_log(self, results: List[Dict], errors: List[Dict]):
        """Save processing log to file"""
        log_data = {
            "processing_date": datetime.now().isoformat(),
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "sample_rate": self.sample_rate,
            "total_files": len(results) + len(errors),
            "successful": len(results),
            "errors": len(errors),
            "results": results,
            "errors": errors,
        }
        
        log_path = self.output_dir / "processing_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing log saved to: {log_path}")


class AudioClassifier:
    """
    Classifies Isan Pin music based on audio features
    """
    
    def __init__(self):
        """Initialize the audio classifier"""
        self.audio_utils = AudioUtils()
        self.text_processor = TextProcessor()
        logger.info("Initialized AudioClassifier")
    
    def classify(self, audio: np.ndarray, sample_rate: int = None) -> Dict:
        """
        Classify audio into Isan Pin music categories
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with classification results
        """
        if sample_rate is None:
            sample_rate = DATA_CONFIG["sample_rate"]
        
        try:
            # Extract audio features
            features = self._extract_classification_features(audio, sample_rate)
            
            # Classify based on features
            classification = self._classify_features(features)
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying audio: {e}")
            return {
                "style": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {e}",
            }
    
    def _extract_classification_features(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Extract features for classification"""
        features = {}
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        features["tempo"] = float(tempo)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features["mfcc_means"] = np.mean(mfcc, axis=1).tolist()
        features["mfcc_stds"] = np.std(mfcc, axis=1).tolist()
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features["chroma_means"] = np.mean(chroma, axis=1).tolist()
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        features["contrast_means"] = np.mean(contrast, axis=1).tolist()
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        features["tonnetz_means"] = np.mean(tonnetz, axis=1).tolist()
        
        # Rhythm features
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]
        features["tempo_estimated"] = float(tempo)
        
        # Dynamic features
        rms = librosa.feature.rms(y=audio)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))
        
        return features
    
    def _classify_features(self, features: Dict) -> Dict:
        """Classify audio based on extracted features"""
        # This is a simplified rule-based classifier
        # In a real implementation, you would use a trained ML model
        
        tempo = features.get("tempo", 0)
        spectral_centroid = features.get("spectral_centroid_mean", 0)
        zcr = features.get("zcr_mean", 0)
        rms_mean = features.get("rms_mean", 0)
        
        # Rule-based classification
        classification = {
            "style": "unknown",
            "confidence": 0.5,
            "reasoning": "Rule-based classification",
        }
        
        # Style classification based on tempo and spectral features
        if tempo < 80:
            classification["style"] = "lam_plearn"
            classification["confidence"] = 0.7
            classification["reasoning"] = f"Slow tempo ({tempo:.1f} BPM) suggests Lam Plearn style"
        elif tempo > 120:
            classification["style"] = "lam_sing"
            classification["confidence"] = 0.7
            classification["reasoning"] = f"Fast tempo ({tempo:.1f} BPM) suggests Lam Sing style"
        elif 80 <= tempo <= 120:
            if spectral_centroid > 2000:
                classification["style"] = "lam_klorn"
                classification["confidence"] = 0.6
                classification["reasoning"] = f"Medium tempo with high spectral centroid suggests Lam Klorn"
            else:
                classification["style"] = "lam_tad"
                classification["confidence"] = 0.6
                classification["reasoning"] = f"Medium tempo with lower spectral centroid suggests Lam Tad"
        
        # Add mood estimation
        if rms_mean < 0.05:
            classification["mood"] = "sad"
            classification["mood_confidence"] = 0.6
        elif rms_mean > 0.15:
            classification["mood"] = "fun"
            classification["mood_confidence"] = 0.6
        else:
            classification["mood"] = "contemplative"
            classification["mood_confidence"] = 0.5
        
        # Add tempo category
        if tempo < 80:
            classification["tempo_category"] = "slow"
        elif tempo > 120:
            classification["tempo_category"] = "fast"
        else:
            classification["tempo_category"] = "medium"
        
        # Add key estimation (simplified)
        # This would be more sophisticated in a real implementation
        classification["estimated_key"] = "D"  # Common key for Isan music
        classification["key_confidence"] = 0.4
        
        return classification


if __name__ == "__main__":
    # Example usage
    collector = DataCollector(
        input_dir="./raw_audio",
        output_dir="./processed_data",
        sample_rate=48000,
    )
    
    # Add classifier
    classifier = AudioClassifier()
    collector.classifier = classifier
    
    # Process all files
    stats = collector.process_all()
    print(f"Processing completed: {stats}")