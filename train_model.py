"""
TinyML Person Detection - Training Pipeline

Complete training script with:
- Custom data loader for preprocessed numpy arrays
- Model compilation and training
- Learning rate scheduling
- Early stopping
- Accuracy evaluation
- Model saving (.h5 format)
- Training history visualization

Tested on GPU machines. Works on CPU but slower.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Import our model architecture
from model_architecture import create_model


class TinyMLDataLoader:
    """
    Data loader for preprocessed numpy arrays
    
    Loads from directory structure:
    preprocessed_dataset/
        train/
            person/*.npy
            no_person/*.npy
        val/
            person/*.npy
            no_person/*.npy
        test/
            person/*.npy
            no_person/*.npy
    """
    
    def __init__(self, dataset_dir, batch_size=32):
        """
        Args:
            dataset_dir: Path to preprocessed dataset
            batch_size: Batch size for training
        """
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        
        # Load metadata
        with open(self.dataset_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.input_shape = tuple(self.metadata['config']['input_shape'])
        
        print(f"Dataset loaded from: {self.dataset_dir}")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {self.batch_size}")
    
    def load_split(self, split='train'):
        """
        Load a dataset split (train/val/test)
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            X: numpy array of images, shape (N, H, W, C)
            y: numpy array of labels, shape (N,), 0=no_person, 1=person
        """
        split_dir = self.dataset_dir / split
        
        X_list = []
        y_list = []
        
        # Class 0: no_person
        no_person_dir = split_dir / 'no_person'
        if no_person_dir.exists():
            for npy_file in no_person_dir.glob('*.npy'):
                img = np.load(npy_file)
                X_list.append(img)
                y_list.append(0)
        
        # Class 1: person
        person_dir = split_dir / 'person'
        if person_dir.exists():
            for npy_file in person_dir.glob('*.npy'):
                img = np.load(npy_file)
                X_list.append(img)
                y_list.append(1)
        
        # Convert to numpy arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"\n{split.upper()} set:")
        print(f"  Total images: {len(X)}")
        print(f"  no_person: {np.sum(y == 0)}")
        print(f"  person: {np.sum(y == 1)}")
        print(f"  Shape: {X.shape}")
        print(f"  Label shape: {y.shape}")
        
        return X, y
    
    def create_tf_dataset(self, X, y, shuffle=True, augment=False):
        """
        Create TensorFlow dataset for efficient training
        
        Args:
            X: numpy array of images
            y: numpy array of labels
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply additional augmentation (not used, already done)
        
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class TinyMLTrainer:
    """Training pipeline for TinyML person detector"""
    
    def __init__(self, 
                 dataset_dir,
                 model_size='auto',
                 batch_size=32,
                 learning_rate=0.001,
                 output_dir='training_output'):
        """
        Args:
            dataset_dir: Path to preprocessed dataset
            model_size: 'tiny', 'small', 'medium', or 'auto'
            batch_size: Training batch size
            learning_rate: Initial learning rate
            output_dir: Directory to save models and logs
        """
        self.dataset_dir = dataset_dir
        self.model_size = model_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data loader
        self.data_loader = TinyMLDataLoader(dataset_dir, batch_size)
        
        # Model
        self.model = None
        self.history = None
        
        # Training metadata
        self.training_start_time = None
        self.training_end_time = None
    
    def build_model(self):
        """Build and compile model"""
        print("\n" + "="*70)
        print("BUILDING MODEL")
        print("="*70)
        
        self.model = create_model(
            input_shape=self.data_loader.input_shape,
            model_size=self.model_size
        )
        
        return self.model
    
    def train(self, epochs=50, patience=10):
        """
        Train the model
        
        Args:
            epochs: Maximum number of training epochs
            patience: Early stopping patience (stop if no improvement)
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load datasets
        X_train, y_train = self.data_loader.load_split('train')
        X_val, y_val = self.data_loader.load_split('val')
        
        # Create TF datasets
        train_dataset = self.data_loader.create_tf_dataset(
            X_train, y_train, shuffle=True
        )
        val_dataset = self.data_loader.create_tf_dataset(
            X_val, y_val, shuffle=False
        )
        
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Early stopping patience: {patience}")
        
        # Callbacks
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(self.output_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train
        self.training_start_time = datetime.now()
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Training duration: {training_duration}")
        print(f"Final training accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        print(f"Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)
        
        # Load test data
        X_test, y_test = self.data_loader.load_split('test')
        test_dataset = self.data_loader.create_tf_dataset(
            X_test, y_test, shuffle=False
        )
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=1)
        
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Per-class accuracy
        y_pred = self.model.predict(test_dataset, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Confusion matrix
        no_person_correct = np.sum((y_test == 0) & (y_pred_classes == 0))
        no_person_total = np.sum(y_test == 0)
        person_correct = np.sum((y_test == 1) & (y_pred_classes == 1))
        person_total = np.sum(y_test == 1)
        
        print(f"\nPer-class accuracy:")
        print(f"  no_person: {no_person_correct}/{no_person_total} = {no_person_correct/no_person_total:.4f}")
        print(f"  person:    {person_correct}/{person_total} = {person_correct/person_total:.4f}")
        
        # Save evaluation results
        eval_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'no_person_accuracy': float(no_person_correct / no_person_total),
            'person_accuracy': float(person_correct / person_total),
            'confusion_matrix': {
                'true_no_person_pred_no_person': int(no_person_correct),
                'true_no_person_pred_person': int(no_person_total - no_person_correct),
                'true_person_pred_no_person': int(person_total - person_correct),
                'true_person_pred_person': int(person_correct)
            }
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n✓ Evaluation results saved to {self.output_dir / 'evaluation_results.json'}")
        
        return test_accuracy
    
    def save_model(self, filename='person_detector.h5'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path = self.output_dir / filename
        self.model.save(model_path)
        
        print(f"\n✓ Model saved to {model_path}")
        
        return model_path
    
    def plot_training_history(self):
        """Plot training curves"""
        if self.history is None:
            raise ValueError("No training history. Call train() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150)
        print(f"\n✓ Training history plot saved to {plot_path}")
        
        plt.close()
    
    def save_training_summary(self):
        """Save training summary as JSON"""
        if self.history is None:
            raise ValueError("No training history. Call train() first.")
        
        summary = {
            'model_config': {
                'model_size': self.model_size,
                'input_shape': self.data_loader.input_shape,
                'total_parameters': int(self.model.count_params())
            },
            'training_config': {
                'batch_size': self.batch_size,
                'initial_learning_rate': self.learning_rate,
                'epochs_trained': len(self.history.history['loss']),
                'training_duration_seconds': (
                    self.training_end_time - self.training_start_time
                ).total_seconds() if self.training_end_time else None
            },
            'final_metrics': {
                'train_accuracy': float(self.history.history['accuracy'][-1]),
                'val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'train_loss': float(self.history.history['loss'][-1]),
                'val_loss': float(self.history.history['val_loss'][-1])
            },
            'best_metrics': {
                'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
                'best_val_loss': float(min(self.history.history['val_loss']))
            }
        }
        
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Training summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train TinyML person detection model'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to preprocessed dataset directory'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='auto',
        choices=['tiny', 'small', 'medium', 'auto'],
        help='Model size (default: auto)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum number of epochs (default: 50)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='training_output',
        help='Output directory for models and logs (default: training_output)'
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("="*70)
    print("SYSTEM INFO")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print("="*70)
    
    # Initialize trainer
    trainer = TinyMLTrainer(
        dataset_dir=args.dataset_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(epochs=args.epochs, patience=args.patience)
    
    # Evaluate
    trainer.evaluate()
    
    # Save model
    trainer.save_model()
    
    # Plot and save training history
    trainer.plot_training_history()
    trainer.save_training_summary()
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"All outputs saved to: {args.output_dir}")
    print(f"Next step: Convert model to TensorFlow Lite with quantization")


if __name__ == '__main__':
    main()
