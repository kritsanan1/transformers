"""
Inference System for Isan Pin AI

This module implements:
- High-level inference pipeline for music generation
- Batch processing capabilities
- Audio post-processing and enhancement
- Quality assessment and filtering
- Export in various formats
- Integration with classification and generation models
"""

import os
import json
import logging
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from ..config import MODEL_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG
from ..models.classification import IsanPinClassifier
from ..models.musicgen import IsanPinMusicGen
from ..data.preprocessing import AudioPreprocessor
from ..utils.audio import AudioUtils

logger = logging.getLogger(__name__)


class IsanPinInference:
    """Main inference class for Isan Pin music generation"""
    
    def __init__(
        self,
        classifier_path: str = None,
        generator_path: str = None,
        device: str = "auto",
        cache_dir: str = None,
    ):
        """
        Initialize inference system
        
        Args:
            classifier_path: Path to trained classifier
            generator_path: Path to fine-tuned generator
            device: Device to use
            cache_dir: Cache directory
        """
        self.device = self._get_device(device)
        self.cache_dir = cache_dir
        self.preprocessor = AudioPreprocessor()
        
        # Initialize models
        self.classifier = None
        self.generator = None
        
        if classifier_path and os.path.exists(classifier_path):
            self.load_classifier(classifier_path)
        
        if generator_path and os.path.exists(generator_path):
            self.load_generator(generator_path)
        else:
            # Load base generator
            self.generator = IsanPinMusicGen(cache_dir=cache_dir)
        
        # Audio utilities
        self.audio_utils = AudioUtils()
        
        logger.info("Isan Pin inference system initialized")
    
    def _get_device(self, device: str) -> torch.device:
        """Get torch device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_classifier(self, path: str):
        """Load the classifier model"""
        try:
            self.classifier = IsanPinClassifier(model_path=path, device=str(self.device))
            logger.info(f"Loaded classifier from {path}")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            self.classifier = None
    
    def load_generator(self, path: str):
        """Load the generator model"""
        try:
            self.generator = IsanPinMusicGen(model_path=path)
            logger.info(f"Loaded generator from {path}")
        except Exception as e:
            logger.error(f"Failed to load generator: {e}")
            self.generator = None
    
    def generate_music(
        self,
        description: str,
        style: str = None,
        mood: str = None,
        duration: float = 30.0,
        num_samples: int = 1,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        post_process: bool = True,
        quality_filter: bool = True,
    ) -> List[Dict]:
        """
        Generate Isan Pin music from description
        
        Args:
            description: Text description
            style: Target style (lam_plearn, lam_sing, etc.)
            mood: Mood descriptor
            duration: Duration in seconds
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            guidance_scale: Guidance scale
            post_process: Whether to apply post-processing
            quality_filter: Whether to filter by quality
            
        Returns:
            List of generated music results
        """
        if not self.generator:
            raise ValueError("Generator model not loaded")
        
        logger.info(f"Generating music: {description}")
        
        results = []
        
        try:
            # Generate audio
            generated_audios = self.generator.generate(
                description=description,
                style=style,
                mood=mood,
                duration=duration,
                num_samples=num_samples,
                temperature=temperature,
                guidance_scale=guidance_scale,
            )
            
            # Process each generated sample
            for i, audio in enumerate(generated_audios):
                result = {
                    'audio': audio,
                    'description': description,
                    'style': style,
                    'mood': mood,
                    'duration': len(audio) / self.preprocessor.sample_rate,
                    'sample_rate': self.preprocessor.sample_rate,
                    'generation_params': {
                        'temperature': temperature,
                        'guidance_scale': guidance_scale,
                    },
                    'quality_score': None,
                    'classification': None,
                }
                
                # Post-processing
                if post_process:
                    audio_processed = self.post_process_audio(audio)
                    result['audio_processed'] = audio_processed
                
                # Quality assessment
                if quality_filter:
                    quality_score = self.assess_quality(audio)
                    result['quality_score'] = quality_score
                
                # Classification
                if self.classifier:
                    try:
                        classification = self.classifier.predict(audio)
                        result['classification'] = classification
                    except Exception as e:
                        logger.warning(f"Classification failed for sample {i}: {e}")
                
                results.append(result)
            
            # Filter by quality if requested
            if quality_filter:
                results = self._filter_by_quality(results)
            
            logger.info(f"Generated {len(results)} music samples")
            return results
            
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
            raise
    
    def batch_generate(
        self,
        descriptions: List[str],
        styles: List[str] = None,
        moods: List[str] = None,
        duration: float = 30.0,
        num_samples_per_description: int = 1,
        max_workers: int = 4,
        post_process: bool = True,
    ) -> List[List[Dict]]:
        """
        Generate music for multiple descriptions in parallel
        
        Args:
            descriptions: List of descriptions
            styles: List of styles (optional)
            moods: List of moods (optional)
            duration: Duration per sample
            num_samples_per_description: Samples per description
            max_workers: Maximum parallel workers
            post_process: Whether to post-process
            
        Returns:
            List of results for each description
        """
        logger.info(f"Batch generating music for {len(descriptions)} descriptions")
        
        # Prepare parameters
        if styles is None:
            styles = [None] * len(descriptions)
        if moods is None:
            moods = [None] * len(descriptions)
        
        # Generate in parallel
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for desc, style, mood in zip(descriptions, styles, moods):
                future = executor.submit(
                    self.generate_music,
                    description=desc,
                    style=style,
                    mood=mood,
                    duration=duration,
                    num_samples=num_samples_per_description,
                    post_process=post_process,
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    results = future.result()
                    all_results.append(results)
                except Exception as e:
                    logger.error(f"Batch generation failed: {e}")
                    all_results.append([])
        
        logger.info(f"Batch generation completed: {len(all_results)} description sets")
        return all_results
    
    def style_transfer(
        self,
        audio: Union[str, Path, np.ndarray],
        target_style: str,
        description: str = None,
        strength: float = 0.7,
        post_process: bool = True,
        save_path: str = None,
    ) -> Dict:
        """
        Apply style transfer to audio
        
        Args:
            audio: Input audio
            target_style: Target Isan Pin style
            description: Optional description
            strength: Transfer strength
            post_process: Whether to post-process
            save_path: Path to save result
            
        Returns:
            Style transfer results
        """
        if not self.generator:
            raise ValueError("Generator model not loaded")
        
        logger.info(f"Applying style transfer to {target_style} style")
        
        try:
            # Apply style transfer
            result_audio = self.generator.style_transfer(
                audio=audio,
                target_style=target_style,
                description=description,
                strength=strength,
            )
            
            result = {
                'audio': result_audio,
                'original_audio': audio if isinstance(audio, np.ndarray) else None,
                'target_style': target_style,
                'strength': strength,
                'description': description,
                'duration': len(result_audio) / self.preprocessor.sample_rate,
                'quality_score': None,
                'classification': None,
            }
            
            # Post-processing
            if post_process:
                audio_processed = self.post_process_audio(result_audio)
                result['audio_processed'] = audio_processed
            
            # Quality assessment
            quality_score = self.assess_quality(result_audio)
            result['quality_score'] = quality_score
            
            # Classification
            if self.classifier:
                try:
                    classification = self.classifier.predict(result_audio)
                    result['classification'] = classification
                except Exception as e:
                    logger.warning(f"Classification failed: {e}")
            
            # Save if path provided
            if save_path:
                sf.write(save_path, result_audio, self.preprocessor.sample_rate)
                logger.info(f"Saved style-transferred audio: {save_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Style transfer failed: {e}")
            raise
    
    def interpolate_styles(
        self,
        description: str,
        style1: str,
        style2: str,
        num_steps: int = 5,
        duration: float = 30.0,
        save_path: str = None,
    ) -> List[Dict]:
        """
        Create style interpolations between two Isan Pin styles
        
        Args:
            description: Base description
            style1: First style
            style2: Second style
            num_steps: Number of interpolation steps
            duration: Duration per sample
            save_path: Path to save results
            
        Returns:
            List of interpolation results
        """
        if not self.generator:
            raise ValueError("Generator model not loaded")
        
        logger.info(f"Creating interpolations between {style1} and {style2}")
        
        results = []
        
        for i in range(num_steps):
            interpolation_factor = i / (num_steps - 1)
            
            try:
                # Generate interpolated audio
                interpolated_audio = self.generator.interpolate_styles(
                    description=description,
                    style1=style1,
                    style2=style2,
                    interpolation_factor=interpolation_factor,
                    num_samples=1,
                )[0]
                
                result = {
                    'audio': interpolated_audio,
                    'description': description,
                    'style1': style1,
                    'style2': style2,
                    'interpolation_factor': interpolation_factor,
                    'duration': len(interpolated_audio) / self.preprocessor.sample_rate,
                    'quality_score': None,
                    'classification': None,
                }
                
                # Quality assessment
                quality_score = self.assess_quality(interpolated_audio)
                result['quality_score'] = quality_score
                
                # Classification
                if self.classifier:
                    try:
                        classification = self.classifier.predict(interpolated_audio)
                        result['classification'] = classification
                    except Exception as e:
                        logger.warning(f"Classification failed for interpolation {i}: {e}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Interpolation step {i} failed: {e}")
                continue
        
        # Save if path provided
        if save_path:
            self._save_interpolations(results, save_path)
        
        logger.info(f"Created {len(results)} style interpolations")
        return results
    
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to generated audio
        
        Args:
            audio: Input audio
            
        Returns:
            Processed audio
        """
        try:
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Apply gentle compression
            audio = self._apply_compression(audio)
            
            # Apply gentle EQ
            audio = self._apply_eq(audio)
            
            # Remove silence at beginning and end
            audio = self._trim_silence(audio)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return audio
    
    def assess_quality(self, audio: np.ndarray) -> float:
        """
        Assess the quality of generated audio
        
        Args:
            audio: Audio signal
            
        Returns:
            Quality score (0.0-1.0)
        """
        try:
            # Basic quality metrics
            features = self.preprocessor.extract_features(audio, feature_type="basic")
            
            # Check for silence
            rms = features.get('rms', 0)
            if rms < 0.01:  # Very quiet
                return 0.1
            
            # Check for clipping
            if np.any(np.abs(audio) > 0.95):
                return 0.3
            
            # Check duration
            duration = features.get('duration', 0)
            if duration < 1.0:  # Too short
                return 0.2
            
            # Spectral analysis
            spectral_features = self.preprocessor.extract_features(audio, feature_type="spectral")
            spectral_centroid = spectral_features.get('spectral_centroid_mean', 0)
            
            # Isan Pin music typically has moderate spectral centroid
            if spectral_centroid < 500 or spectral_centroid > 8000:
                return 0.4
            
            # Combined quality score
            quality_score = 0.7  # Base score
            
            # Adjust based on RMS
            if rms > 0.1:
                quality_score += 0.2
            
            # Adjust based on spectral features
            if 1000 < spectral_centroid < 5000:
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Neutral score
    
    def export_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        format: str = "wav",
        sample_rate: int = None,
        metadata: Dict = None,
    ) -> str:
        """
        Export audio to file
        
        Args:
            audio: Audio signal
            output_path: Output file path
            format: Audio format
            sample_rate: Sample rate
            metadata: Metadata to embed
            
        Returns:
            Path to exported file
        """
        if sample_rate is None:
            sample_rate = self.preprocessor.sample_rate
        
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format.lower() == "wav":
                sf.write(output_path, audio, sample_rate)
            elif format.lower() == "mp3":
                # Convert to mp3 using pydub (if available)
                try:
                    from pydub import AudioSegment
                    
                    # First save as wav
                    temp_wav = output_path.replace('.mp3', '_temp.wav')
                    sf.write(temp_wav, audio, sample_rate)
                    
                    # Convert to mp3
                    audio_segment = AudioSegment.from_wav(temp_wav)
                    audio_segment.export(output_path, format="mp3")
                    
                    # Remove temp file
                    os.remove(temp_wav)
                    
                except ImportError:
                    logger.warning("pydub not available, saving as WAV")
                    sf.write(output_path.replace('.mp3', '.wav'), audio, sample_rate)
            else:
                sf.write(output_path, audio, sample_rate)
            
            # Add metadata if provided
            if metadata and format.lower() == "wav":
                self._add_metadata(output_path, metadata)
            
            logger.info(f"Exported audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio export failed: {e}")
            raise
    
    def _filter_by_quality(self, results: List[Dict], min_quality: float = 0.5) -> List[Dict]:
        """Filter results by quality score"""
        filtered = []
        for result in results:
            quality_score = result.get('quality_score', 0)
            if quality_score is None or quality_score >= min_quality:
                filtered.append(result)
        return filtered
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle compression to audio"""
        # Simple compression implementation
        threshold = 0.7
        ratio = 3.0
        
        # Apply soft compression
        compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )
        
        return compressed
    
    def _apply_eq(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle EQ to enhance Isan Pin characteristics"""
        # Simple high-pass filter to remove low-frequency noise
        try:
            from scipy import signal
            sos = signal.butter(4, 80, btype='high', fs=self.preprocessor.sample_rate, output='sos')
            filtered = signal.sosfilt(sos, audio)
            return filtered
        except ImportError:
            return audio
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end"""
        # Find non-silent regions
        non_silent = np.where(np.abs(audio) > threshold)[0]
        
        if len(non_silent) > 0:
            start_idx = max(0, non_silent[0] - int(0.1 * self.preprocessor.sample_rate))
            end_idx = min(len(audio), non_silent[-1] + int(0.1 * self.preprocessor.sample_rate))
            return audio[start_idx:end_idx]
        
        return audio
    
    def _save_interpolations(self, results: List[Dict], base_path: str):
        """Save interpolation results"""
        base_path = Path(base_path)
        output_dir = base_path.parent
        
        for i, result in enumerate(results):
            interpolation_factor = result['interpolation_factor']
            audio = result['audio']
            
            # Create filename with interpolation factor
            filename = f"{base_path.stem}_interp_{interpolation_factor:.2f}{base_path.suffix}"
            output_path = output_dir / filename
            
            sf.write(output_path, audio, self.preprocessor.sample_rate)
            logger.info(f"Saved interpolation: {output_path}")
    
    def _add_metadata(self, audio_path: str, metadata: Dict):
        """Add metadata to audio file"""
        try:
            # This is a simplified implementation
            # In practice, you'd use a library like mutagen for proper metadata handling
            
            # Create a companion JSON file with metadata
            json_path = audio_path.replace('.wav', '_metadata.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved metadata: {json_path}")
            
        except Exception as e:
            logger.warning(f"Failed to add metadata: {e}")


def create_inference_system(
    classifier_path: str = None,
    generator_path: str = None,
    cache_dir: str = None,
) -> IsanPinInference:
    """
    Create inference system with default configuration
    
    Args:
        classifier_path: Path to trained classifier
        generator_path: Path to fine-tuned generator
        cache_dir: Cache directory
        
    Returns:
        Inference system
    """
    logger.info("Creating Isan Pin inference system...")
    
    inference = IsanPinInference(
        classifier_path=classifier_path,
        generator_path=generator_path,
        cache_dir=cache_dir,
    )
    
    return inference


if __name__ == "__main__":
    # Example usage
    inference = IsanPinInference()
    
    # Test quality assessment
    test_audio = np.random.randn(48000) * 0.1  # Low-amplitude noise
    
    try:
        quality_score = inference.assess_quality(test_audio)
        print(f"Quality score for test audio: {quality_score:.3f}")
        
        # Test post-processing
        processed_audio = inference.post_process_audio(test_audio)
        print(f"Post-processed audio shape: {processed_audio.shape}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("Isan Pin inference system loaded successfully!")