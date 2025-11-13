"""
Audio Preprocessing Module for Isan Pin AI

This module handles:
- Audio format standardization
- Noise reduction and filtering
- Data augmentation
- Feature extraction
- Dataset creation for training
"""

import os
import json
import random
import logging
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from ..config import DATA_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG
from ..utils.audio import AudioUtils

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Preprocesses audio data for machine learning training
    """
    
    def __init__(
        self,
        sample_rate: int = None,
        n_fft: int = None,
        hop_length: int = None,
        n_mels: int = None,
    ):
        """
        Initialize the audio preprocessor
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate or DATA_CONFIG["sample_rate"]
        self.n_fft = n_fft or AUDIO_CONFIG["n_fft"]
        self.hop_length = hop_length or AUDIO_CONFIG["hop_length"]
        self.n_mels = n_mels or AUDIO_CONFIG["n_mels"]
        
        self.audio_utils = AudioUtils()
        
        logger.info(f"Initialized AudioPreprocessor: sr={self.sample_rate}, n_fft={self.n_fft}")
    
    def standardize_audio(self, audio: np.ndarray, target_sr: int = None) -> np.ndarray:
        """
        Standardize audio format
        
        Args:
            audio: Input audio signal
            target_sr: Target sample rate (uses default if None)
            
        Returns:
            Standardized audio signal
        """
        target_sr = target_sr or self.sample_rate
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Resample if necessary
        if target_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def extract_features(self, audio: np.ndarray, feature_type: str = "all") -> Dict:
        """
        Extract audio features for analysis and classification
        
        Args:
            audio: Audio signal
            feature_type: Type of features to extract ('all', 'basic', 'spectral', 'temporal')
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        if feature_type in ["all", "basic"]:
            # Basic features
            features["duration"] = len(audio) / self.sample_rate
            features["mean"] = float(np.mean(audio))
            features["std"] = float(np.std(audio))
            features["rms"] = float(np.sqrt(np.mean(audio ** 2)))
            features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        
        if feature_type in ["all", "spectral"]:
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
            features["spectral_centroid_std"] = float(np.std(spectral_centroid))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            features["spectral_contrast_mean"] = np.mean(spectral_contrast, axis=1).tolist()
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features["mfcc_means"] = np.mean(mfcc, axis=1).tolist()
            features["mfcc_stds"] = np.std(mfcc, axis=1).tolist()
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features["chroma_means"] = np.mean(chroma, axis=1).tolist()
            
        if feature_type in ["all", "temporal"]:
            # Temporal features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features["tempo"] = float(tempo)
            features["beats_per_second"] = len(beats) / features["duration"] if features.get("duration") else 0
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            features["onset_strength_mean"] = float(np.mean(onset_env))
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))
        
        return features
    
    def create_spectrogram(self, audio: np.ndarray, save_path: str = None) -> np.ndarray:
        """
        Create mel-spectrogram from audio
        
        Args:
            audio: Audio signal
            save_path: Optional path to save the spectrogram image
            
        Returns:
            Mel-spectrogram as numpy array
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save as image if requested
        if save_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(mel_spec_db, sr=self.sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return mel_spec_db
    
    def apply_noise_reduction(self, audio: np.ndarray, noise_reduction_factor: float = 0.5) -> np.ndarray:
        """
        Apply simple noise reduction
        
        Args:
            audio: Input audio
            noise_reduction_factor: Strength of noise reduction (0.0-1.0)
            
        Returns:
            Noise-reduced audio
        """
        # Simple spectral subtraction noise reduction
        # This is a basic implementation - for production, use more sophisticated methods
        
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise profile from first few frames
            noise_profile = np.mean(magnitude[:, :5], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            magnitude_denoised = magnitude - noise_reduction_factor * noise_profile
            magnitude_denoised = np.maximum(magnitude_denoised, 0)
            
            # Reconstruct signal
            stft_denoised = magnitude_denoised * np.exp(1j * phase)
            audio_denoised = librosa.istft(stft_denoised, hop_length=self.hop_length)
            
            return audio_denoised
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}. Returning original audio.")
            return audio
    
    def apply_filtering(self, audio: np.ndarray, filter_type: str = "highpass", cutoff_freq: float = 80.0) -> np.ndarray:
        """
        Apply audio filtering
        
        Args:
            audio: Input audio
            filter_type: Type of filter ('highpass', 'lowpass', 'bandpass')
            cutoff_freq: Cutoff frequency in Hz
            
        Returns:
            Filtered audio
        """
        try:
            from scipy import signal
            
            # Design filter
            nyquist = self.sample_rate / 2
            
            if filter_type == "highpass":
                sos = signal.butter(4, cutoff_freq / nyquist, btype='high', output='sos')
            elif filter_type == "lowpass":
                sos = signal.butter(4, cutoff_freq / nyquist, btype='low', output='sos')
            elif filter_type == "bandpass":
                low_cutoff = cutoff_freq[0] / nyquist
                high_cutoff = cutoff_freq[1] / nyquist
                sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', output='sos')
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            # Apply filter
            filtered_audio = signal.sosfilt(sos, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Filtering failed: {e}. Returning original audio.")
            return audio
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str = "pitch_shift", **kwargs) -> np.ndarray:
        """
        Apply data augmentation to audio
        
        Args:
            audio: Input audio
            augmentation_type: Type of augmentation
            **kwargs: Additional parameters for augmentation
            
        Returns:
            Augmented audio
        """
        try:
            if augmentation_type == "pitch_shift":
                n_steps = kwargs.get("n_steps", random.uniform(-2, 2))
                return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
            elif augmentation_type == "time_stretch":
                rate = kwargs.get("rate", random.uniform(0.9, 1.1))
                return librosa.effects.time_stretch(audio, rate=rate)
            
            elif augmentation_type == "add_noise":
                noise_factor = kwargs.get("noise_factor", random.uniform(0.01, 0.05))
                noise = np.random.normal(0, noise_factor, len(audio))
                return audio + noise
            
            elif augmentation_type == "volume_change":
                gain = kwargs.get("gain", random.uniform(0.7, 1.3))
                return audio * gain
            
            elif augmentation_type == "reverb":
                # Simple reverb effect
                delay_samples = int(kwargs.get("delay_ms", 100) * self.sample_rate / 1000)
                decay = kwargs.get("decay", 0.5)
                
                reverb = np.zeros_like(audio)
                if len(audio) > delay_samples:
                    reverb[delay_samples:] = audio[:-delay_samples] * decay
                
                return audio + reverb
            
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}")
                
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}. Returning original audio.")
            return audio


class DatasetCreator:
    """
    Creates training datasets for Isan Pin AI
    """
    
    def __init__(
        self,
        output_dir: str,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        """
        Initialize dataset creator
        
        Args:
            output_dir: Output directory for dataset
            preprocessor: Audio preprocessor instance
        """
        self.output_dir = Path(output_dir)
        self.preprocessor = preprocessor or AudioPreprocessor()
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DatasetCreator: output={output_dir}")
    
    def create_dataset(
        self,
        audio_files: List[Path],
        labels: Optional[List[str]] = None,
        test_size: float = 0.15,
        val_size: float = 0.05,
        random_state: int = 42,
        augment_train: bool = True,
    ) -> Dict:
        """
        Create training dataset from audio files
        
        Args:
            audio_files: List of audio file paths
            labels: Optional labels for supervised learning
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            augment_train: Whether to augment training data
            
        Returns:
            Dictionary with dataset information
        """
        logger.info(f"Creating dataset from {len(audio_files)} audio files")
        
        # Split data
        if labels is not None:
            # Stratified split if labels are provided
            X_train, X_temp, y_train, y_temp = train_test_split(
                audio_files, labels, test_size=(test_size + val_size), 
                random_state=random_state, stratify=labels
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=test_size/(test_size + val_size),
                random_state=random_state, stratify=y_temp
            )
        else:
            # Random split
            X_train, X_temp = train_test_split(
                audio_files, test_size=(test_size + val_size), 
                random_state=random_state
            )
            X_val, X_test = train_test_split(
                X_temp, test_size=test_size/(test_size + val_size),
                random_state=random_state
            )
            y_train = y_val = y_test = None
        
        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Process and save each split
        splits = [
            ("train", X_train, y_train, self.train_dir, augment_train),
            ("validation", X_val, y_val, self.val_dir, False),
            ("test", X_test, y_test, self.test_dir, False),
        ]
        
        dataset_info = {}
        
        for split_name, files, labels, output_dir, augment in splits:
            logger.info(f"Processing {split_name} split...")
            
            split_info = self._process_split(
                files, labels, output_dir, augment, split_name
            )
            
            dataset_info[split_name] = split_info
        
        # Save dataset metadata
        metadata = {
            "created_date": pd.Timestamp.now().isoformat(),
            "total_files": len(audio_files),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "test_percentage": test_size * 100,
            "val_percentage": val_size * 100,
            "augment_train": augment_train,
            "sample_rate": self.preprocessor.sample_rate,
            "splits": dataset_info,
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset created successfully: {self.output_dir}")
        return metadata
    
    def _process_split(
        self,
        files: List[Path],
        labels: Optional[List[str]],
        output_dir: Path,
        augment: bool,
        split_name: str,
    ) -> Dict:
        """Process a single data split"""
        
        processed_files = []
        augment_files = []
        
        for i, file_path in enumerate(files):
            try:
                # Load and preprocess audio
                audio, sr = librosa.load(file_path, sr=self.preprocessor.sample_rate, mono=True)
                audio = self.preprocessor.standardize_audio(audio)
                
                # Save original
                output_filename = f"{split_name}_{i:06d}.wav"
                output_path = output_dir / output_filename
                sf.write(output_path, audio, self.preprocessor.sample_rate)
                
                file_info = {
                    "filename": output_filename,
                    "original_path": str(file_path),
                    "duration": len(audio) / self.preprocessor.sample_rate,
                    "sample_rate": self.preprocessor.sample_rate,
                }
                
                if labels is not None:
                    file_info["label"] = labels[i]
                
                processed_files.append(file_info)
                
                # Apply augmentation if enabled
                if augment:
                    augment_files.extend(
                        self._augment_file(audio, output_dir, split_name, i, file_info)
                    )
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        split_info = {
            "files": processed_files,
            "augmented_files": augment_files,
            "total_original": len(processed_files),
            "total_augmented": len(augment_files),
            "total_files": len(processed_files) + len(augment_files),
        }
        
        return split_info
    
    def _augment_file(
        self,
        audio: np.ndarray,
        output_dir: Path,
        split_name: str,
        index: int,
        original_info: Dict,
    ) -> List[Dict]:
        """Apply data augmentation to a file"""
        
        augmented_files = []
        augmentation_types = ["pitch_shift", "time_stretch", "add_noise"]
        
        for aug_type in augmentation_types:
            try:
                # Apply augmentation
                augmented_audio = self.preprocessor.augment_audio(audio, aug_type)
                
                # Save augmented version
                aug_filename = f"{split_name}_{index:06d}_{aug_type}.wav"
                aug_path = output_dir / aug_filename
                sf.write(aug_path, augmented_audio, self.preprocessor.sample_rate)
                
                aug_info = {
                    "filename": aug_filename,
                    "original_index": index,
                    "augmentation": aug_type,
                    "original_file": original_info["filename"],
                    "duration": len(augmented_audio) / self.preprocessor.sample_rate,
                }
                
                if "label" in original_info:
                    aug_info["label"] = original_info["label"]
                
                augmented_files.append(aug_info)
                
            except Exception as e:
                logger.warning(f"Augmentation {aug_type} failed for index {index}: {e}")
        
        return augmented_files
    
    def create_metadata_csv(self, dataset_dir: str = None) -> str:
        """
        Create CSV metadata file for the dataset
        
        Args:
            dataset_dir: Dataset directory (uses self.output_dir if None)
            
        Returns:
            Path to the created CSV file
        """
        dataset_dir = dataset_dir or self.output_dir
        dataset_path = Path(dataset_dir)
        
        all_metadata = []
        
        # Process each split
        for split_name in ["train", "validation", "test"]:
            split_dir = dataset_path / split_name
            metadata_file = dataset_path / f"{split_name}_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    split_data = json.load(f)
                
                # Add split information to each file
                for file_info in split_data["files"]:
                    file_info["split"] = split_name
                    file_info["augmented"] = False
                    all_metadata.append(file_info)
                
                # Add augmented files if they exist
                if "augmented_files" in split_data:
                    for aug_info in split_data["augmented_files"]:
                        aug_info["split"] = split_name
                        aug_info["augmented"] = True
                        all_metadata.append(aug_info)
        
        # Create DataFrame
        df = pd.DataFrame(all_metadata)
        
        # Save to CSV
        csv_path = dataset_path / "dataset_metadata.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"Metadata CSV created: {csv_path}")
        return str(csv_path)


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioPreprocessor()
    
    # Test audio preprocessing
    test_audio = np.random.randn(48000)  # 1 second of noise
    
    # Extract features
    features = preprocessor.extract_features(test_audio)
    print("Extracted features:", list(features.keys()))
    
    # Create spectrogram
    spec = preprocessor.create_spectrogram(test_audio)
    print("Spectrogram shape:", spec.shape)
    
    # Test augmentation
    augmented = preprocessor.augment_audio(test_audio, "pitch_shift", n_steps=2)
    print("Augmentation successful")