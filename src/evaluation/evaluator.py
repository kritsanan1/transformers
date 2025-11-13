"""
Evaluation and Quality Assessment System for Isan Pin AI

This module implements:
- Comprehensive evaluation metrics for generated music
- Comparison with reference Isan Pin music
- Style consistency assessment
- Audio quality metrics
- Cultural authenticity evaluation
- Automated evaluation pipeline
- Human evaluation interface
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import torch

from ..config import MODEL_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG
from ..data.preprocessing import AudioPreprocessor
from ..models.classification import IsanPinClassifier
from ..utils.audio import AudioUtils

logger = logging.getLogger(__name__)


class IsanPinEvaluator:
    """Comprehensive evaluator for Isan Pin music generation"""
    
    def __init__(
        self,
        reference_audio_dir: str = None,
        classifier_path: str = None,
        device: str = "auto",
    ):
        """
        Initialize evaluator
        
        Args:
            reference_audio_dir: Directory with reference Isan Pin audio
            classifier_path: Path to trained classifier
            device: Device to use
        """
        self.device = self._get_device(device)
        self.reference_audio_dir = Path(reference_audio_dir) if reference_audio_dir else None
        self.preprocessor = AudioPreprocessor()
        
        # Load classifier if available
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            try:
                self.classifier = IsanPinClassifier(model_path=classifier_path, device=str(self.device))
                logger.info("Loaded classifier for evaluation")
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")
        
        # Load reference data if available
        self.reference_features = None
        if self.reference_audio_dir and self.reference_audio_dir.exists():
            self.reference_features = self._load_reference_features()
        
        # Audio utilities
        self.audio_utils = AudioUtils()
        
        logger.info("Isan Pin evaluator initialized")
    
    def _get_device(self, device: str) -> torch.device:
        """Get torch device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_reference_features(self) -> Dict:
        """Load and extract features from reference audio"""
        logger.info("Loading reference features...")
        
        reference_features = {
            'global': {},
            'by_style': {},
            'by_mood': {},
        }
        
        if not self.reference_audio_dir.exists():
            logger.warning("Reference audio directory not found")
            return reference_features
        
        all_features = []
        
        # Process all audio files in reference directory
        for audio_file in self.reference_audio_dir.rglob("*.wav"):
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.preprocessor.sample_rate, mono=True)
                
                # Extract features
                features = self.preprocessor.extract_features(audio, feature_type="all")
                features['file_path'] = str(audio_file)
                features['duration'] = len(audio) / sr
                
                # Extract style and mood from filename or metadata
                # This is a simplified approach - in practice, you'd use proper metadata
                filename = audio_file.stem.lower()
                style = self._extract_style_from_filename(filename)
                mood = self._extract_mood_from_filename(filename)
                
                features['style'] = style
                features['mood'] = mood
                
                all_features.append(features)
                
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                continue
        
        # Calculate statistics
        if all_features:
            reference_features['global'] = self._calculate_feature_statistics(all_features)
            
            # Group by style
            style_groups = {}
            for features in all_features:
                style = features.get('style', 'unknown')
                if style not in style_groups:
                    style_groups[style] = []
                style_groups[style].append(features)
            
            for style, group_features in style_groups.items():
                reference_features['by_style'][style] = self._calculate_feature_statistics(group_features)
            
            # Group by mood
            mood_groups = {}
            for features in all_features:
                mood = features.get('mood', 'unknown')
                if mood not in mood_groups:
                    mood_groups[mood] = []
                mood_groups[mood].append(features)
            
            for mood, group_features in mood_groups.items():
                reference_features['by_mood'][mood] = self._calculate_feature_statistics(group_features)
        
        logger.info(f"Loaded reference features from {len(all_features)} audio files")
        return reference_features
    
    def _extract_style_from_filename(self, filename: str) -> str:
        """Extract style from filename (simplified)"""
        for style in MODEL_CONFIG["style_definitions"].keys():
            if style in filename:
                return style
        return 'unknown'
    
    def _extract_mood_from_filename(self, filename: str) -> str:
        """Extract mood from filename (simplified)"""
        mood_keywords = ['sad', 'happy', 'contemplative', 'romantic', 'energetic', 'calm']
        for mood in mood_keywords:
            if mood in filename:
                return mood
        return 'unknown'
    
    def _calculate_feature_statistics(self, features_list: List[Dict]) -> Dict:
        """Calculate statistics from a list of feature dictionaries"""
        if not features_list:
            return {}
        
        # Collect all numeric features
        numeric_features = {}
        for features in features_list:
            for key, value in features.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_features:
                        numeric_features[key] = []
                    numeric_features[key].append(value)
        
        # Calculate statistics
        statistics = {}
        for key, values in numeric_features.items():
            if values:
                statistics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                }
        
        return statistics
    
    def evaluate_single_audio(
        self,
        audio: Union[str, Path, np.ndarray],
        reference_style: str = None,
        reference_mood: str = None,
        detailed: bool = True,
    ) -> Dict:
        """
        Evaluate a single audio file
        
        Args:
            audio: Audio file path or signal
            reference_style: Expected style for comparison
            reference_mood: Expected mood for comparison
            detailed: Whether to include detailed metrics
            
        Returns:
            Evaluation results
        """
        try:
            # Load audio if path is provided
            if isinstance(audio, (str, Path)):
                audio_signal, sr = librosa.load(audio, sr=self.preprocessor.sample_rate, mono=True)
            else:
                audio_signal = audio
            
            # Extract features
            features = self.preprocessor.extract_features(audio_signal, feature_type="all")
            
            # Basic quality metrics
            quality_metrics = self._assess_basic_quality(audio_signal)
            
            # Style consistency
            style_consistency = self._assess_style_consistency(audio_signal, reference_style)
            
            # Audio quality
            audio_quality = self._assess_audio_quality(audio_signal)
            
            # Cultural authenticity
            cultural_authenticity = self._assess_cultural_authenticity(audio_signal, reference_style)
            
            # Reference comparison
            reference_similarity = None
            if self.reference_features and (reference_style or reference_mood):
                reference_similarity = self._compare_with_reference(
                    features, reference_style, reference_mood
                )
            
            # Classification consistency
            classification_consistency = None
            if self.classifier and reference_style:
                classification_consistency = self._assess_classification_consistency(
                    audio_signal, reference_style
                )
            
            # Compile results
            results = {
                'basic_metrics': quality_metrics,
                'style_consistency': style_consistency,
                'audio_quality': audio_quality,
                'cultural_authenticity': cultural_authenticity,
                'reference_similarity': reference_similarity,
                'classification_consistency': classification_consistency,
                'overall_score': None,
                'detailed': detailed,
            }
            
            # Calculate overall score
            results['overall_score'] = self._calculate_overall_score(results)
            
            # Add detailed features if requested
            if detailed:
                results['extracted_features'] = features
                results['duration'] = len(audio_signal) / self.preprocessor.sample_rate
                results['sample_rate'] = self.preprocessor.sample_rate
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'error': str(e),
                'overall_score': 0.0,
            }
    
    def _assess_basic_quality(self, audio: np.ndarray) -> Dict:
        """Assess basic audio quality metrics"""
        metrics = {}
        
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        metrics['rms'] = float(rms)
        
        # Dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        metrics['dynamic_range'] = float(dynamic_range)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        metrics['zero_crossing_rate'] = float(zcr)
        
        # Check for silence
        is_silent = rms < 0.01
        metrics['is_silent'] = bool(is_silent)
        
        # Check for clipping
        is_clipping = np.any(np.abs(audio) > 0.95)
        metrics['is_clipping'] = bool(is_clipping)
        
        # Duration
        duration = len(audio) / self.preprocessor.sample_rate
        metrics['duration'] = float(duration)
        
        # Quality score (0-1)
        quality_score = 1.0
        if is_silent:
            quality_score *= 0.1
        if is_clipping:
            quality_score *= 0.5
        if duration < 1.0:
            quality_score *= 0.7
        if rms < 0.05:
            quality_score *= 0.8
        
        metrics['quality_score'] = float(quality_score)
        
        return metrics
    
    def _assess_style_consistency(self, audio: np.ndarray, reference_style: str = None) -> Dict:
        """Assess consistency with expected Isan Pin style"""
        features = self.preprocessor.extract_features(audio, feature_type="spectral")
        
        consistency_metrics = {}
        
        # Tempo analysis
        tempo = features.get('tempo', 120)  # Default tempo
        
        # Isan Pin tempo ranges by style
        style_tempo_ranges = {
            'lam_plearn': (60, 90),
            'lam_sing': (120, 160),
            'lam_klorn': (80, 120),
            'lam_tad': (100, 140),
            'lam_puen': (90, 130),
        }
        
        if reference_style and reference_style in style_tempo_ranges:
            min_tempo, max_tempo = style_tempo_ranges[reference_style]
            tempo_consistency = 1.0 if min_tempo <= tempo <= max_tempo else 0.5
        else:
            # General Isan Pin tempo range
            tempo_consistency = 1.0 if 60 <= tempo <= 160 else 0.7
        
        consistency_metrics['tempo_consistency'] = float(tempo_consistency)
        
        # Spectral characteristics
        spectral_centroid = features.get('spectral_centroid_mean', 1000)
        spectral_rolloff = features.get('spectral_rolloff_mean', 5000)
        
        # Isan Pin typically has moderate spectral characteristics
        if 500 <= spectral_centroid <= 5000:
            spectral_consistency = 1.0
        elif 300 <= spectral_centroid <= 8000:
            spectral_consistency = 0.8
        else:
            spectral_consistency = 0.5
        
        consistency_metrics['spectral_consistency'] = float(spectral_consistency)
        
        # MFCC consistency (simplified)
        mfcc_means = features.get('mfcc_means', [0] * 13)
        if mfcc_means[0] > -10:  # First MFCC coefficient typically high for traditional music
            mfcc_consistency = 0.9
        else:
            mfcc_consistency = 0.7
        
        consistency_metrics['mfcc_consistency'] = float(mfcc_consistency)
        
        # Overall style consistency
        overall_consistency = np.mean([
            tempo_consistency,
            spectral_consistency,
            mfcc_consistency
        ])
        
        consistency_metrics['overall_consistency'] = float(overall_consistency)
        
        return consistency_metrics
    
    def _assess_audio_quality(self, audio: np.ndarray) -> Dict:
        """Assess technical audio quality"""
        quality_metrics = {}
        
        # Signal-to-noise ratio (simplified)
        # Estimate noise as the minimum amplitude
        noise_level = np.percentile(np.abs(audio), 10)
        signal_level = np.percentile(np.abs(audio), 90)
        
        if noise_level > 0:
            snr = 20 * np.log10(signal_level / noise_level)
        else:
            snr = 40  # Assume good SNR if no noise detected
        
        quality_metrics['snr_db'] = float(snr)
        
        # Harmonic distortion (simplified)
        # This is a very basic harmonic analysis
        try:
            # Compute spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.preprocessor.sample_rate)
            spectral_centroid_mean = np.mean(spectral_centroid)
            
            # High spectral centroid might indicate distortion
            if spectral_centroid_mean > 8000:
                distortion_score = 0.6
            elif spectral_centroid_mean > 5000:
                distortion_score = 0.8
            else:
                distortion_score = 1.0
            
            quality_metrics['distortion_score'] = float(distortion_score)
            
        except Exception:
            quality_metrics['distortion_score'] = 0.8
        
        # Frequency balance
        try:
            # Compute spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.preprocessor.sample_rate)
            rolloff_mean = np.mean(rolloff)
            
            # Balance score based on rolloff
            if 2000 <= rolloff_mean <= 8000:
                balance_score = 1.0
            elif 1000 <= rolloff_mean <= 12000:
                balance_score = 0.8
            else:
                balance_score = 0.6
            
            quality_metrics['frequency_balance'] = float(balance_score)
            
        except Exception:
            quality_metrics['frequency_balance'] = 0.8
        
        # Overall audio quality score
        audio_quality = np.mean([
            min(snr / 40, 1.0),  # Normalize SNR
            quality_metrics.get('distortion_score', 0.8),
            quality_metrics.get('frequency_balance', 0.8)
        ])
        
        quality_metrics['audio_quality_score'] = float(audio_quality)
        
        return quality_metrics
    
    def _assess_cultural_authenticity(self, audio: np.ndarray, reference_style: str = None) -> Dict:
        """Assess cultural authenticity for Isan Pin music"""
        authenticity_metrics = {}
        
        # Extract features
        features = self.preprocessor.extract_features(audio, feature_type="all")
        
        # Tempo authenticity
        tempo = features.get('tempo', 120)
        
        # Traditional Isan Pin tempos
        if reference_style:
            style_tempo_ranges = {
                'lam_plearn': (60, 90),
                'lam_sing': (120, 160),
                'lam_klorn': (80, 120),
                'lam_tad': (100, 140),
                'lam_puen': (90, 130),
            }
            if reference_style in style_tempo_ranges:
                min_tempo, max_tempo = style_tempo_ranges[reference_style]
                tempo_authenticity = 1.0 if min_tempo <= tempo <= max_tempo else 0.6
            else:
                tempo_authenticity = 0.8
        else:
            # General authenticity (moderate tempo)
            tempo_authenticity = 1.0 if 80 <= tempo <= 140 else 0.7
        
        authenticity_metrics['tempo_authenticity'] = float(tempo_authenticity)
        
        # Rhythmic complexity (zero crossing rate as proxy)
        zcr = features.get('zero_crossing_rate', 0.1)
        # Traditional music often has moderate rhythmic complexity
        if 0.05 <= zcr <= 0.2:
            rhythmic_authenticity = 1.0
        elif 0.02 <= zcr <= 0.3:
            rhythmic_authenticity = 0.8
        else:
            rhythmic_authenticity = 0.6
        
        authenticity_metrics['rhythmic_authenticity'] = float(rhythmic_authenticity)
        
        # Spectral authenticity
        spectral_centroid = features.get('spectral_centroid_mean', 1000)
        # Traditional Isan Pin has moderate spectral content
        if 500 <= spectral_centroid <= 3000:
            spectral_authenticity = 1.0
        elif 300 <= spectral_centroid <= 5000:
            spectral_authenticity = 0.8
        else:
            spectral_authenticity = 0.6
        
        authenticity_metrics['spectral_authenticity'] = float(spectral_authenticity)
        
        # Overall cultural authenticity
        cultural_authenticity = np.mean([
            tempo_authenticity,
            rhythmic_authenticity,
            spectral_authenticity
        ])
        
        authenticity_metrics['cultural_authenticity_score'] = float(cultural_authenticity)
        
        return authenticity_metrics
    
    def _compare_with_reference(
        self,
        features: Dict,
        reference_style: str = None,
        reference_mood: str = None,
    ) -> Dict:
        """Compare features with reference data"""
        if not self.reference_features:
            return {'similarity_score': 0.5, 'reference_available': False}
        
        # Select reference group
        if reference_style and reference_style in self.reference_features['by_style']:
            reference_stats = self.reference_features['by_style'][reference_style]
        elif reference_mood and reference_mood in self.reference_features['by_mood']:
            reference_stats = self.reference_features['by_mood'][reference_mood]
        else:
            reference_stats = self.reference_features['global']
        
        if not reference_stats:
            return {'similarity_score': 0.5, 'reference_available': False}
        
        # Compare features
        similarities = []
        
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)) and feature_name in reference_stats:
                ref_mean = reference_stats[feature_name].get('mean', 0)
                ref_std = reference_stats[feature_name].get('std', 1)
                
                if ref_std > 0:
                    # Z-score normalization
                    z_score = abs(feature_value - ref_mean) / ref_std
                    # Convert to similarity (0-1)
                    similarity = max(0, 1 - z_score / 3)  # 3 standard deviations = 0 similarity
                    similarities.append(similarity)
        
        # Average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.5
        
        return {
            'similarity_score': float(avg_similarity),
            'reference_available': True,
            'num_features_compared': len(similarities),
        }
    
    def _assess_classification_consistency(
        self,
        audio: np.ndarray,
        reference_style: str = None,
    ) -> Dict:
        """Assess consistency with classifier predictions"""
        if not self.classifier or not reference_style:
            return {'consistency_score': 0.5, 'classifier_available': False}
        
        try:
            prediction = self.classifier.predict(audio)
            predicted_style = prediction.get('predicted_style', 'unknown')
            confidence = prediction.get('confidence', 0)
            
            # Check if predicted style matches expected style
            if predicted_style == reference_style:
                consistency = confidence
            else:
                # Partial consistency if top-3 contains expected style
                top_3 = prediction.get('top_3_predictions', [])
                expected_in_top3 = any(pred.get('style') == reference_style for pred in top_3)
                consistency = 0.3 if expected_in_top3 else 0.1
            
            return {
                'consistency_score': float(consistency),
                'predicted_style': predicted_style,
                'expected_style': reference_style,
                'confidence': float(confidence),
                'classifier_available': True,
            }
            
        except Exception as e:
            logger.warning(f"Classification consistency assessment failed: {e}")
            return {'consistency_score': 0.5, 'classifier_available': True, 'error': str(e)}
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall evaluation score"""
        scores = []
        
        # Basic quality score
        if 'basic_metrics' in results and results['basic_metrics']:
            scores.append(results['basic_metrics'].get('quality_score', 0.5))
        
        # Style consistency score
        if 'style_consistency' in results and results['style_consistency']:
            scores.append(results['style_consistency'].get('overall_consistency', 0.5))
        
        # Audio quality score
        if 'audio_quality' in results and results['audio_quality']:
            scores.append(results['audio_quality'].get('audio_quality_score', 0.5))
        
        # Cultural authenticity score
        if 'cultural_authenticity' in results and results['cultural_authenticity']:
            scores.append(results['cultural_authenticity'].get('cultural_authenticity_score', 0.5))
        
        # Reference similarity score
        if 'reference_similarity' in results and results['reference_similarity']:
            scores.append(results['reference_similarity'].get('similarity_score', 0.5))
        
        # Classification consistency score
        if 'classification_consistency' in results and results['classification_consistency']:
            scores.append(results['classification_consistency'].get('consistency_score', 0.5))
        
        # Calculate weighted average
        if scores:
            return float(np.mean(scores))
        else:
            return 0.5
    
    def evaluate_dataset(
        self,
        audio_files: List[Union[str, Path]],
        labels: List[str] = None,
        styles: List[str] = None,
        moods: List[str] = None,
        save_report: str = None,
    ) -> Dict:
        """
        Evaluate a dataset of audio files
        
        Args:
            audio_files: List of audio file paths
            labels: Optional labels for supervised evaluation
            styles: Optional expected styles
            moods: Optional expected moods
            save_report: Path to save evaluation report
            
        Returns:
            Dataset evaluation results
        """
        logger.info(f"Evaluating dataset with {len(audio_files)} audio files")
        
        all_results = []
        
        for i, audio_file in enumerate(audio_files):
            logger.info(f"Evaluating file {i+1}/{len(audio_files)}: {audio_file}")
            
            # Get expected style/mood if provided
            expected_style = styles[i] if styles and i < len(styles) else None
            expected_mood = moods[i] if moods and i < len(moods) else None
            
            # Evaluate single file
            result = self.evaluate_single_audio(
                audio=audio_file,
                reference_style=expected_style,
                reference_mood=expected_mood,
                detailed=False,  # Faster evaluation for large datasets
            )
            
            result['file_path'] = str(audio_file)
            result['expected_style'] = expected_style
            result['expected_mood'] = expected_mood
            result['index'] = i
            
            all_results.append(result)
        
        # Calculate dataset statistics
        dataset_stats = self._calculate_dataset_statistics(all_results)
        
        # Compile final results
        final_results = {
            'individual_results': all_results,
            'dataset_statistics': dataset_stats,
            'total_files': len(audio_files),
            'evaluation_date': datetime.now().isoformat(),
        }
        
        # Save report if requested
        if save_report:
            self._save_evaluation_report(final_results, save_report)
        
        logger.info(f"Dataset evaluation completed. Average score: {dataset_stats.get('mean_score', 0):.3f}")
        return final_results
    
    def _calculate_dataset_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics for dataset evaluation"""
        scores = []
        style_accuracies = []
        
        for result in results:
            overall_score = result.get('overall_score', 0)
            scores.append(overall_score)
            
            # Check style accuracy if available
            if 'classification_consistency' in result:
                consistency = result['classification_consistency']
                if consistency and consistency.get('classifier_available'):
                    predicted = consistency.get('predicted_style')
                    expected = consistency.get('expected_style')
                    if predicted and expected:
                        style_accuracies.append(1.0 if predicted == expected else 0.0)
        
        statistics = {
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'std_score': float(np.std(scores)) if scores else 0.0,
            'min_score': float(np.min(scores)) if scores else 0.0,
            'max_score': float(np.max(scores)) if scores else 0.0,
            'median_score': float(np.median(scores)) if scores else 0.0,
            'num_files': len(results),
            'style_accuracy': float(np.mean(style_accuracies)) if style_accuracies else None,
        }
        
        return statistics
    
    def _save_evaluation_report(self, results: Dict, output_path: str):
        """Save evaluation report to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save text summary
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Isan Pin AI Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            
            stats = results['dataset_statistics']
            f.write(f"Total Files: {stats['num_files']}\n")
            f.write(f"Mean Score: {stats['mean_score']:.3f}\n")
            f.write(f"Std Score: {stats['std_score']:.3f}\n")
            f.write(f"Min Score: {stats['min_score']:.3f}\n")
            f.write(f"Max Score: {stats['max_score']:.3f}\n")
            
            if stats['style_accuracy'] is not None:
                f.write(f"Style Accuracy: {stats['style_accuracy']:.3f}\n")
            
            f.write(f"\nEvaluation Date: {results['evaluation_date']}\n")
        
        logger.info(f"Evaluation report saved: {json_path} and {txt_path}")
    
    def create_visualization(
        self,
        evaluation_results: Dict,
        output_path: str = None,
        show_plots: bool = False,
    ) -> str:
        """
        Create visualization of evaluation results
        
        Args:
            evaluation_results: Results from evaluate_dataset
            output_path: Path to save visualization
            show_plots: Whether to show plots interactively
            
        Returns:
            Path to saved visualization
        """
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Extract scores
            scores = []
            for result in evaluation_results['individual_results']:
                score = result.get('overall_score', 0)
                scores.append(score)
            
            if not scores:
                logger.warning("No scores available for visualization")
                return None
            
            # Score distribution
            axes[0, 0].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Score Distribution')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            axes[0, 0].legend()
            
            # Score over files
            axes[0, 1].plot(scores, marker='o', alpha=0.6)
            axes[0, 1].set_title('Scores Over Files')
            axes[0, 1].set_xlabel('File Index')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Statistics summary
            stats = evaluation_results['dataset_statistics']
            categories = ['Mean', 'Std', 'Min', 'Max']
            values = [stats['mean_score'], stats['std_score'], stats['min_score'], stats['max_score']]
            
            axes[1, 0].bar(categories, values, color=['green', 'orange', 'red', 'blue'], alpha=0.7)
            axes[1, 0].set_title('Statistics Summary')
            axes[1, 0].set_ylabel('Score')
            for i, v in enumerate(values):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # Style accuracy if available
            if stats['style_accuracy'] is not None:
                axes[1, 1].bar(['Style Accuracy'], [stats['style_accuracy']], color='purple', alpha=0.7)
                axes[1, 1].set_title('Style Accuracy')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].text(0, stats['style_accuracy'] + 0.02, f"{stats['style_accuracy']:.3f}", ha='center')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Style Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Style Accuracy')
            
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved: {output_path}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None


def create_evaluator(
    reference_audio_dir: str = None,
    classifier_path: str = None,
) -> IsanPinEvaluator:
    """
    Create evaluator with default configuration
    
    Args:
        reference_audio_dir: Directory with reference audio
        classifier_path: Path to trained classifier
        
    Returns:
        Isan Pin evaluator
    """
    logger.info("Creating Isan Pin evaluator...")
    
    evaluator = IsanPinEvaluator(
        reference_audio_dir=reference_audio_dir,
        classifier_path=classifier_path,
    )
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    evaluator = IsanPinEvaluator()
    
    # Test evaluation
    test_audio = np.random.randn(48000) * 0.1  # Low-amplitude noise
    
    try:
        results = evaluator.evaluate_single_audio(test_audio, reference_style="lam_plearn")
        print(f"Overall score: {results.get('overall_score', 0):.3f}")
        print(f"Basic quality score: {results.get('basic_metrics', {}).get('quality_score', 0):.3f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    print("Isan Pin evaluator loaded successfully!")