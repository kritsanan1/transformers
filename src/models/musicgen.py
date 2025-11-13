"""
MusicGen Fine-tuning for Isan Pin Music Generation

This module implements:
- Facebook MusicGen model loading and configuration
- Fine-tuning on Isan Pin music dataset
- Text-to-music generation with Thai language support
- Style conditioning for traditional Thai music
- Model persistence and loading
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import soundfile as sf
import librosa
from datetime import datetime

from ..config import MODEL_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG
from ..data.preprocessing import AudioPreprocessor
from ..utils.audio import AudioUtils

logger = logging.getLogger(__name__)


class IsanPinMusicDataset(Dataset):
    """Dataset for Isan Pin music generation training"""
    
    def __init__(
        self,
        audio_paths: List[Path],
        descriptions: List[str],
        preprocessor: AudioPreprocessor,
        max_length: int = 30,  # seconds
        max_description_length: int = 512,
    ):
        """
        Initialize dataset
        
        Args:
            audio_paths: List of audio file paths
            descriptions: List of text descriptions
            preprocessor: Audio preprocessor instance
            max_length: Maximum audio length in seconds
            max_description_length: Maximum description length
        """
        self.audio_paths = audio_paths
        self.descriptions = descriptions
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.max_description_length = max_description_length
        
        # Filter out invalid samples
        self.valid_indices = self._filter_valid_samples()
    
    def _filter_valid_samples(self) -> List[int]:
        """Filter out invalid samples"""
        valid = []
        for i, (audio_path, desc) in enumerate(zip(self.audio_paths, self.descriptions)):
            if audio_path.exists() and desc and len(desc) > 0:
                valid.append(i)
        return valid
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a single item"""
        original_idx = self.valid_indices[idx]
        audio_path = self.audio_paths[original_idx]
        description = self.descriptions[original_idx]
        
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=self.preprocessor.sample_rate, mono=True)
            audio = self.preprocessor.standardize_audio(audio)
            
            # Trim or pad to max_length
            target_length = int(self.max_length * sr)
            if len(audio) > target_length:
                # Random crop
                start_idx = np.random.randint(0, len(audio) - target_length)
                audio = audio[start_idx:start_idx + target_length]
            else:
                # Pad with zeros
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Convert to mel-spectrogram for conditioning
            mel_spec = self.preprocessor.create_spectrogram(audio)
            
            # Truncate or pad description
            words = description.split()
            if len(words) > self.max_description_length // 4:  # Rough estimate
                description = " ".join(words[:self.max_description_length // 4])
            
            return {
                'audio': torch.FloatTensor(audio),
                'mel_spec': torch.FloatTensor(mel_spec),
                'description': description,
                'audio_path': str(audio_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return dummy data
            return {
                'audio': torch.zeros(int(self.max_length * sr)),
                'mel_spec': torch.zeros((128, 1292)),
                'description': "Traditional Isan Pin music",
                'audio_path': str(audio_path)
            }


class IsanPinMusicGen:
    """MusicGen model fine-tuned for Isan Pin music generation"""
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        model_path: str = None,
        device: str = "auto",
        cache_dir: str = None,
    ):
        """
        Initialize MusicGen model
        
        Args:
            model_name: Base MusicGen model name
            model_path: Path to fine-tuned model
            device: Device to use
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        self.preprocessor = AudioPreprocessor()
        
        # Load base model and processor
        logger.info(f"Loading base MusicGen model: {model_name}")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self.model.to(self.device)
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
        
        # Load fine-tuned model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info("Loaded fine-tuned model")
        
        # Initialize tokenizer for Thai text
        self._setup_tokenizer()
    
    def _get_device(self, device: str) -> torch.device:
        """Get torch device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _setup_tokenizer(self):
        """Setup tokenizer for Thai text processing"""
        try:
            # Try to use a multilingual tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium",
                cache_dir=self.cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Using default.")
            self.tokenizer = None
    
    def prepare_description(self, description: str, style: str = None, mood: str = None) -> str:
        """
        Prepare description for generation
        
        Args:
            description: Base description
            style: Isan Pin style (lam_plearn, lam_sing, etc.)
            mood: Mood descriptor
            
        Returns:
            Processed description
        """
        # Add style-specific information
        if style:
            style_desc = MODEL_CONFIG["style_definitions"].get(style, {})
            thai_name = style_desc.get("thai_name", style)
            mood_list = style_desc.get("mood", [])
            
            description = f"{thai_name} style. {description}"
            
            if mood and mood in mood_list:
                description = f"{description} {mood} mood."
        
        # Add cultural context
        cultural_context = "Traditional Northeastern Thai Isan Pin music. "
        description = f"{cultural_context}{description}"
        
        # Ensure it's in English for the model
        # In a full implementation, you'd translate from Thai to English
        
        return description
    
    def fine_tune(
        self,
        train_dataset: IsanPinMusicDataset,
        val_dataset: IsanPinMusicDataset,
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        save_path: str = None,
        gradient_accumulation_steps: int = 4,
    ) -> Dict:
        """
        Fine-tune the model on Isan Pin music
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            save_path: Path to save the best model
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Training history
        """
        logger.info("Starting fine-tuning...")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Set up training
        self.model.train()
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
        }
        
        logger.info(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Process descriptions
                    descriptions = [
                        self.prepare_description(desc) for desc in batch['description']
                    ]
                    
                    # Tokenize descriptions
                    inputs = self.processor(
                        text=descriptions,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Prepare audio inputs
                    audio_inputs = batch['audio'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=audio_inputs)
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    # Update weights
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    train_loss += loss.item() * gradient_accumulation_steps
                    num_train_batches += 1
                    
                    if (batch_idx + 1) % 100 == 0:
                        logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                                   f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
            
            # Validation phase
            val_loss = self._validate_finetune(val_loader)
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch + 1
                
                if save_path:
                    self.save_model(save_path)
        
        logger.info(f"Fine-tuning completed. Best validation loss: {history['best_val_loss']:.4f} "
                   f"at epoch {history['best_epoch']}")
        
        return history
    
    def _validate_finetune(self, val_loader: DataLoader) -> float:
        """Validate during fine-tuning"""
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Process descriptions
                    descriptions = [
                        self.prepare_description(desc) for desc in batch['description']
                    ]
                    
                    # Tokenize descriptions
                    inputs = self.processor(
                        text=descriptions,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Prepare audio inputs
                    audio_inputs = batch['audio'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=audio_inputs)
                    loss = outputs.loss
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
                
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        return total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
    
    def generate(
        self,
        description: str,
        style: str = None,
        mood: str = None,
        duration: float = 30.0,
        num_samples: int = 1,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        save_path: str = None,
    ) -> List[np.ndarray]:
        """
        Generate Isan Pin music from text description
        
        Args:
            description: Text description of the music
            style: Isan Pin style (lam_plearn, lam_sing, etc.)
            mood: Mood descriptor
            duration: Duration in seconds
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            guidance_scale: Guidance scale for generation
            save_path: Path to save generated audio
            
        Returns:
            List of generated audio arrays
        """
        logger.info(f"Generating Isan Pin music: {description}")
        
        try:
            # Prepare description
            processed_description = self.prepare_description(description, style, mood)
            logger.info(f"Processed description: {processed_description}")
            
            # Tokenize description
            inputs = self.processor(
                text=processed_description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate audio
            self.model.eval()
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=int(duration * 50),  # Rough estimate
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                    num_return_sequences=num_samples,
                )
            
            # Convert to numpy arrays
            generated_audios = []
            for i, audio_tensor in enumerate(audio_values):
                audio_array = audio_tensor.cpu().numpy()
                generated_audios.append(audio_array)
                
                # Save if path provided
                if save_path:
                    base_path = Path(save_path)
                    if num_samples > 1:
                        save_file = base_path.parent / f"{base_path.stem}_{i+1}{base_path.suffix}"
                    else:
                        save_file = base_path
                    
                    sf.write(save_file, audio_array, self.preprocessor.sample_rate)
                    logger.info(f"Saved generated audio: {save_file}")
            
            logger.info(f"Successfully generated {num_samples} audio samples")
            return generated_audios
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def style_transfer(
        self,
        audio: Union[str, Path, np.ndarray],
        target_style: str,
        description: str = None,
        strength: float = 0.7,
        save_path: str = None,
    ) -> np.ndarray:
        """
        Apply style transfer to audio
        
        Args:
            audio: Input audio (path or array)
            target_style: Target Isan Pin style
            description: Optional description
            strength: Style transfer strength (0.0-1.0)
            save_path: Path to save result
            
        Returns:
            Style-transferred audio
        """
        logger.info(f"Applying style transfer to {target_style} style")
        
        try:
            # Load audio if path is provided
            if isinstance(audio, (str, Path)):
                audio_signal, sr = librosa.load(audio, sr=self.preprocessor.sample_rate, mono=True)
            else:
                audio_signal = audio
            
            # Prepare description for target style
            if description is None:
                description = f"Music in {target_style} style"
            
            style_description = self.prepare_description(description, target_style)
            
            # Extract features from original audio
            original_features = self.preprocessor.extract_features(audio_signal)
            
            # Generate new audio in target style
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            generated_audio = self.generate(
                description=style_description,
                style=target_style,
                duration=len(audio_signal) / self.preprocessor.sample_rate,
                num_samples=1,
                temperature=0.8,
            )[0]
            
            # Blend original and generated audio
            if strength < 1.0:
                # Ensure same length
                min_len = min(len(audio_signal), len(generated_audio))
                audio_signal = audio_signal[:min_len]
                generated_audio = generated_audio[:min_len]
                
                # Blend
                result_audio = (1 - strength) * audio_signal + strength * generated_audio
            else:
                result_audio = generated_audio
            
            # Save if path provided
            if save_path:
                sf.write(save_path, result_audio, self.preprocessor.sample_rate)
                logger.info(f"Saved style-transferred audio: {save_path}")
            
            return result_audio
            
        except Exception as e:
            logger.error(f"Style transfer failed: {e}")
            raise
    
    def save_model(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'processor_config': self.processor.to_dict() if hasattr(self.processor, 'to_dict') else None,
            'model_name': self.model_name,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def interpolate_styles(
        self,
        description: str,
        style1: str,
        style2: str,
        interpolation_factor: float = 0.5,
        num_samples: int = 1,
        save_path: str = None,
    ) -> List[np.ndarray]:
        """
        Interpolate between two Isan Pin styles
        
        Args:
            description: Base description
            style1: First style
            style2: Second style
            interpolation_factor: Interpolation factor (0.0=style1, 1.0=style2)
            num_samples: Number of samples
            save_path: Path to save results
            
        Returns:
            List of interpolated audio
        """
        logger.info(f"Interpolating between {style1} and {style2} (factor: {interpolation_factor})")
        
        # Generate audio for both styles
        audio1 = self.generate(
            description=description,
            style=style1,
            num_samples=1,
            temperature=0.7,
        )[0]
        
        audio2 = self.generate(
            description=description,
            style=style2,
            num_samples=1,
            temperature=0.7,
        )[0]
        
        # Interpolate
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        interpolated = (1 - interpolation_factor) * audio1 + interpolation_factor * audio2
        
        # Save if path provided
        if save_path:
            sf.write(save_path, interpolated, self.preprocessor.sample_rate)
            logger.info(f"Saved interpolated audio: {save_path}")
        
        return [interpolated]


def create_isan_pin_generator(
    model_name: str = "facebook/musicgen-small",
    model_path: str = None,
    cache_dir: str = None,
    train: bool = False,
    data_dir: str = None,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
) -> IsanPinMusicGen:
    """
    Create and optionally train Isan Pin music generator
    
    Args:
        model_name: Base MusicGen model
        model_path: Path to fine-tuned model
        cache_dir: Cache directory
        train: Whether to fine-tune
        data_dir: Training data directory
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        
    Returns:
        Isan Pin music generator
    """
    logger.info("Creating Isan Pin music generator...")
    
    # Initialize generator
    generator = IsanPinMusicGen(
        model_name=model_name,
        model_path=model_path,
        cache_dir=cache_dir
    )
    
    if train and data_dir:
        logger.info("Preparing training data...")
        
        # This would load actual training data
        # For now, we'll skip the training step
        logger.info("Training functionality requires actual Isan Pin music dataset")
        logger.info("Skipping training step - using base model")
    
    return generator


if __name__ == "__main__":
    # Example usage
    generator = IsanPinMusicGen()
    
    # Test generation
    test_description = "Traditional Isan Pin music with slow tempo and contemplative mood"
    
    try:
        generated_audio = generator.generate(
            description=test_description,
            style="lam_plearn",
            duration=5.0,  # Short duration for testing
            num_samples=1,
        )
        
        print(f"Generated {len(generated_audio)} audio samples")
        print(f"Audio shape: {generated_audio[0].shape}")
        print(f"Duration: {len(generated_audio[0]) / generator.preprocessor.sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Generation failed (expected without fine-tuning): {e}")
    
    print("Isan Pin MusicGen module loaded successfully!")