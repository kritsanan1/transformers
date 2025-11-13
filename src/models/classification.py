"""
Advanced Audio Classification System for Isan Pin Music

This module implements:
- Deep learning models for Isan Pin style classification
- Feature extraction and preprocessing for ML models
- Training and evaluation pipelines
- Model persistence and loading
- Thai music style classification (lam_plearn, lam_sing, lam_klorn, lam_tad, lam_puen)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import joblib
import librosa
import soundfile as sf
from datetime import datetime

from ..config import MODEL_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG
from ..data.preprocessing import AudioPreprocessor
from ..utils.audio import AudioUtils

logger = logging.getLogger(__name__)


class IsanPinDataset(Dataset):
    """Dataset class for Isan Pin music classification"""
    
    def __init__(
        self,
        audio_paths: List[Path],
        labels: List[str],
        preprocessor: AudioPreprocessor,
        max_length: int = 30,  # seconds
        augment: bool = False,
    ):
        """
        Initialize dataset
        
        Args:
            audio_paths: List of audio file paths
            labels: List of corresponding labels
            preprocessor: Audio preprocessor instance
            max_length: Maximum audio length in seconds
            augment: Whether to apply data augmentation
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.augment = augment
        self.label_encoder = LabelEncoder()
        
        # Encode labels
        if labels:
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.encoded_labels = None
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        """Get a single item"""
        audio_path = self.audio_paths[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.preprocessor.sample_rate, mono=True)
            
            # Standardize audio
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
            
            # Apply augmentation if enabled
            if self.augment and np.random.random() > 0.5:
                augmentation_type = np.random.choice(["pitch_shift", "time_stretch", "add_noise"])
                audio = self.preprocessor.augment_audio(audio, augmentation_type)
            
            # Extract features
            features = self.preprocessor.extract_features(audio, feature_type="spectral")
            
            # Convert to tensor
            mel_spec = self.preprocessor.create_spectrogram(audio)
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension
            
            # Get label
            if self.encoded_labels is not None:
                label = torch.LongTensor([self.encoded_labels[idx]])
            else:
                label = torch.LongTensor([0])  # Default label
            
            return mel_spec, label
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return a dummy sample
            dummy_spec = torch.zeros((1, 128, 1292))  # Typical mel-spectrogram size
            dummy_label = torch.LongTensor([0])
            return dummy_spec, dummy_label


class IsanPinCNN(nn.Module):
    """CNN model for Isan Pin music classification"""
    
    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 1,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize CNN model
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout_rate: Dropout rate
        """
        super(IsanPinCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """Forward pass"""
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class IsanPinClassifier:
    """Main classifier for Isan Pin music styles"""
    
    def __init__(
        self,
        model_path: str = None,
        num_classes: int = 5,
        device: str = "auto",
    ):
        """
        Initialize classifier
        
        Args:
            model_path: Path to pre-trained model
            num_classes: Number of output classes
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.num_classes = num_classes
        self.device = self._get_device(device)
        self.preprocessor = AudioPreprocessor()
        
        # Initialize model
        self.model = IsanPinCNN(num_classes=num_classes)
        self.model.to(self.device)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.class_names = list(MODEL_CONFIG["style_definitions"].keys())
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No pre-trained model loaded. Model will be initialized randomly.")
    
    def _get_device(self, device: str) -> torch.device:
        """Get torch device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def train(
        self,
        train_dataset: IsanPinDataset,
        val_dataset: IsanPinDataset,
        num_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        patience: int = 10,
        save_path: str = None,
    ) -> Dict:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            save_path: Path to save the best model
            
        Returns:
            Training history
        """
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
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0,
        }
        
        # Early stopping
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels.squeeze()).sum().item()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                         f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                         f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > history['best_val_acc']:
                history['best_val_acc'] = val_acc
                history['best_epoch'] = epoch + 1
                best_model_state = self.model.state_dict()
                patience_counter = 0
                
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        logger.info(f"Training completed. Best validation accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")
        
        return history
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.squeeze()).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        return avg_val_loss, val_acc
    
    def predict(self, audio: Union[str, Path, np.ndarray]) -> Dict:
        """
        Predict the style of an audio file
        
        Args:
            audio: Audio file path or audio signal
            
        Returns:
            Prediction results
        """
        self.model.eval()
        
        try:
            # Load audio if path is provided
            if isinstance(audio, (str, Path)):
                audio_signal, sr = librosa.load(audio, sr=self.preprocessor.sample_rate, mono=True)
            else:
                audio_signal = audio
            
            # Preprocess
            audio_signal = self.preprocessor.standardize_audio(audio_signal)
            
            # Create mel-spectrogram
            mel_spec = self.preprocessor.create_spectrogram(audio_signal)
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            mel_spec = mel_spec.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(mel_spec)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted.cpu().numpy()[0]
            
            # Create results
            results = {
                'predicted_class': int(predicted_class),
                'predicted_style': self.class_names[predicted_class],
                'confidence': float(probabilities[predicted_class]),
                'all_probabilities': {
                    style: float(prob) for style, prob in zip(self.class_names, probabilities)
                },
                'top_3_predictions': []
            }
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            for idx in top_3_indices:
                results['top_3_predictions'].append({
                    'style': self.class_names[idx],
                    'probability': float(probabilities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def evaluate(self, test_dataset: IsanPinDataset, batch_size: int = 16) -> Dict:
        """
        Evaluate the model on test data
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size
            
        Returns:
            Evaluation results
        """
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.class_names,
            output_dict=True
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': self.class_names,
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        for class_name in self.class_names:
            if class_name in class_report:
                logger.info(f"{class_name}: Precision={class_report[class_name]['precision']:.3f}, "
                           f"Recall={class_report[class_name]['recall']:.3f}, "
                           f"F1={class_report[class_name]['f1-score']:.3f}")
        
        return results
    
    def save_model(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        logger.info(f"Model loaded from {path}")
    
    def cross_validate(self, dataset: IsanPinDataset, cv: int = 5, batch_size: int = 16) -> Dict:
        """
        Perform cross-validation
        
        Args:
            dataset: Dataset to cross-validate
            cv: Number of folds
            batch_size: Batch size
            
        Returns:
            Cross-validation results
        """
        # This is a simplified implementation
        # In practice, you'd use sklearn's cross-validation with a custom scorer
        
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Split dataset into folds
        fold_size = len(dataset) // cv
        fold_accuracies = []
        
        for fold in range(cv):
            # Create train/test split for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < cv - 1 else len(dataset)
            
            test_indices = list(range(test_start, test_end))
            train_indices = [i for i in range(len(dataset)) if i not in test_indices]
            
            # Create subset datasets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)
            
            # Train on this fold
            fold_history = self.train(
                train_subset,
                test_subset,
                num_epochs=10,  # Fewer epochs for CV
                batch_size=batch_size
            )
            
            fold_accuracies.append(fold_history['best_val_acc'])
            logger.info(f"Fold {fold+1}: Validation accuracy = {fold_history['best_val_acc']:.4f}")
        
        results = {
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
        }
        
        logger.info(f"Cross-validation completed: Mean={results['mean_accuracy']:.4f}, "
                   f"Std={results['std_accuracy']:.4f}")
        
        return results


def create_style_classifier(
    data_dir: str,
    model_path: str = None,
    train: bool = True,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
) -> IsanPinClassifier:
    """
    Create and train a style classifier
    
    Args:
        data_dir: Directory containing training data
        model_path: Path to save/load model
        train: Whether to train the model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Trained classifier
    """
    logger.info("Creating Isan Pin style classifier...")
    
    # Initialize classifier
    classifier = IsanPinClassifier(num_classes=5)
    
    if train:
        # Load datasets
        train_dataset = IsanPinDataset(
            audio_paths=[],  # Will be populated from data_dir
            labels=[],
            preprocessor=AudioPreprocessor(),
            augment=True
        )
        
        val_dataset = IsanPinDataset(
            audio_paths=[],
            labels=[],
            preprocessor=AudioPreprocessor(),
            augment=False
        )
        
        # Train the model
        history = classifier.train(
            train_dataset,
            val_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path=model_path
        )
        
        logger.info("Training completed!")
    
    else:
        # Load pre-trained model
        if model_path and os.path.exists(model_path):
            classifier.load_model(model_path)
            logger.info("Loaded pre-trained model")
        else:
            logger.warning("No pre-trained model available")
    
    return classifier


if __name__ == "__main__":
    # Example usage
    classifier = IsanPinClassifier()
    
    # Test prediction
    test_audio = np.random.randn(48000)  # 1 second of noise
    
    try:
        results = classifier.predict(test_audio)
        print("Prediction results:", results)
    except Exception as e:
        print(f"Prediction failed (expected for random noise): {e}")
    
    print("Isan Pin Classifier module loaded successfully!")