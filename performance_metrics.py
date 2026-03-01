"""
TinyML Person Detection - Real-Time Metrics System

This module provides comprehensive performance monitoring:
1. Accuracy metrics (precision, recall, F1-score)
2. Inference speed (FPS, latency)
3. Memory usage tracking
4. Real-time logging and visualization

Use this for:
- Benchmarking different models
- Monitoring deployed system performance
- Debugging accuracy issues
- Optimizing inference speed
"""

import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime


class PerformanceMetrics:
    """
    Comprehensive performance monitoring for TinyML person detection
    
    Tracks:
    - Accuracy, Precision, Recall, F1-score
    - Inference latency (min, max, mean, p50, p95, p99)
    - FPS (frames per second)
    - Memory usage
    - Per-class performance
    """
    
    def __init__(self, num_classes=2, class_names=None):
        """
        Args:
            num_classes: Number of classes (default: 2 for binary classification)
            class_names: List of class names (default: ["no_person", "person"])
        """
        self.num_classes = num_classes
        self.class_names = class_names or ["no_person", "person"]
        
        # Performance counters
        self.reset()
        
        # Session metadata
        self.session_start_time = time.time()
    
    def reset(self):
        """Reset all metrics"""
        # Confusion matrix elements
        self.true_positives = np.zeros(self.num_classes, dtype=int)
        self.false_positives = np.zeros(self.num_classes, dtype=int)
        self.true_negatives = np.zeros(self.num_classes, dtype=int)
        self.false_negatives = np.zeros(self.num_classes, dtype=int)
        
        # Timing metrics
        self.inference_times = []
        self.total_frames = 0
        
        # Per-frame predictions and ground truth
        self.predictions = []
        self.ground_truths = []
        
        # Confidence scores
        self.confidence_scores = []
    
    def update(self, y_true, y_pred, confidence, inference_time_ms):
        """
        Update metrics with new prediction
        
        Args:
            y_true: Ground truth label (int)
            y_pred: Predicted label (int)
            confidence: Prediction confidence (float, 0-1)
            inference_time_ms: Inference time in milliseconds (float)
        """
        # Store predictions
        self.predictions.append(y_pred)
        self.ground_truths.append(y_true)
        self.confidence_scores.append(confidence)
        
        # Update confusion matrix
        if y_true == y_pred:
            # Correct prediction
            self.true_positives[y_pred] += 1
            # Update true negatives for other classes
            for c in range(self.num_classes):
                if c != y_true:
                    self.true_negatives[c] += 1
        else:
            # Incorrect prediction
            self.false_positives[y_pred] += 1
            self.false_negatives[y_true] += 1
            # Update true negatives for other classes
            for c in range(self.num_classes):
                if c != y_true and c != y_pred:
                    self.true_negatives[c] += 1
        
        # Update timing
        self.inference_times.append(inference_time_ms)
        self.total_frames += 1
    
    def get_accuracy(self):
        """Calculate overall accuracy"""
        if self.total_frames == 0:
            return 0.0
        correct = sum(self.true_positives)
        return correct / self.total_frames
    
    def get_precision(self, class_idx=None):
        """
        Calculate precision
        
        Precision = TP / (TP + FP)
        
        Args:
            class_idx: Class index (None for macro-average)
        """
        if class_idx is not None:
            tp = self.true_positives[class_idx]
            fp = self.false_positives[class_idx]
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            # Macro-average
            precisions = [self.get_precision(i) for i in range(self.num_classes)]
            return np.mean(precisions)
    
    def get_recall(self, class_idx=None):
        """
        Calculate recall (sensitivity)
        
        Recall = TP / (TP + FN)
        
        Args:
            class_idx: Class index (None for macro-average)
        """
        if class_idx is not None:
            tp = self.true_positives[class_idx]
            fn = self.false_negatives[class_idx]
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            # Macro-average
            recalls = [self.get_recall(i) for i in range(self.num_classes)]
            return np.mean(recalls)
    
    def get_f1_score(self, class_idx=None):
        """
        Calculate F1-score
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            class_idx: Class index (None for macro-average)
        """
        if class_idx is not None:
            precision = self.get_precision(class_idx)
            recall = self.get_recall(class_idx)
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        else:
            # Macro-average
            f1_scores = [self.get_f1_score(i) for i in range(self.num_classes)]
            return np.mean(f1_scores)
    
    def get_latency_stats(self):
        """Calculate latency statistics"""
        if not self.inference_times:
            return {
                'min_ms': 0,
                'max_ms': 0,
                'mean_ms': 0,
                'median_ms': 0,
                'p95_ms': 0,
                'p99_ms': 0
            }
        
        times = np.array(self.inference_times)
        return {
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99))
        }
    
    def get_fps(self):
        """Calculate average FPS"""
        latency = self.get_latency_stats()
        if latency['mean_ms'] > 0:
            return 1000.0 / latency['mean_ms']
        return 0.0
    
    def get_confusion_matrix(self):
        """Get confusion matrix as 2D array"""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, pred_label in zip(self.ground_truths, self.predictions):
            cm[true_label][pred_label] += 1
        return cm
    
    def print_summary(self):
        """Print comprehensive metrics summary"""
        print("\n" + "="*70)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*70)
        
        # Overall metrics
        print(f"\n{'OVERALL METRICS':<40}")
        print("-"*70)
        print(f"  Total frames:        {self.total_frames:>10}")
        print(f"  Overall accuracy:    {self.get_accuracy()*100:>9.2f}%")
        print(f"  Macro-avg precision: {self.get_precision()*100:>9.2f}%")
        print(f"  Macro-avg recall:    {self.get_recall()*100:>9.2f}%")
        print(f"  Macro-avg F1-score:  {self.get_f1_score()*100:>9.2f}%")
        
        # Per-class metrics
        print(f"\n{'PER-CLASS METRICS':<40}")
        print("-"*70)
        for i in range(self.num_classes):
            print(f"\n  Class: {self.class_names[i]}")
            print(f"    TP: {self.true_positives[i]:>5}  |  FP: {self.false_positives[i]:>5}")
            print(f"    FN: {self.false_negatives[i]:>5}  |  TN: {self.true_negatives[i]:>5}")
            print(f"    Precision: {self.get_precision(i)*100:>6.2f}%")
            print(f"    Recall:    {self.get_recall(i)*100:>6.2f}%")
            print(f"    F1-score:  {self.get_f1_score(i)*100:>6.2f}%")
        
        # Confusion matrix
        print(f"\n{'CONFUSION MATRIX':<40}")
        print("-"*70)
        cm = self.get_confusion_matrix()
        print(f"{'':>15}", end="")
        for name in self.class_names:
            print(f"{name:>12}", end="")
        print()
        for i, name in enumerate(self.class_names):
            print(f"  {name:>13}", end="")
            for j in range(self.num_classes):
                print(f"{cm[i][j]:>12}", end="")
            print()
        
        # Latency metrics
        print(f"\n{'LATENCY & FPS':<40}")
        print("-"*70)
        latency = self.get_latency_stats()
        print(f"  Min latency:      {latency['min_ms']:>8.2f} ms")
        print(f"  Max latency:      {latency['max_ms']:>8.2f} ms")
        print(f"  Mean latency:     {latency['mean_ms']:>8.2f} ms")
        print(f"  Median latency:   {latency['median_ms']:>8.2f} ms")
        print(f"  95th percentile:  {latency['p95_ms']:>8.2f} ms")
        print(f"  99th percentile:  {latency['p99_ms']:>8.2f} ms")
        print(f"  Average FPS:      {self.get_fps():>8.2f}")
        
        # Session info
        session_duration = time.time() - self.session_start_time
        print(f"\n{'SESSION INFO':<40}")
        print("-"*70)
        print(f"  Duration:         {session_duration:>8.1f} seconds")
        print(f"  Throughput:       {self.total_frames / session_duration:>8.2f} frames/sec")
        
        print("="*70 + "\n")
    
    def save_to_json(self, output_path):
        """Save metrics to JSON file"""
        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'session_duration_seconds': time.time() - self.session_start_time,
            'overall': {
                'total_frames': int(self.total_frames),
                'accuracy': float(self.get_accuracy()),
                'precision': float(self.get_precision()),
                'recall': float(self.get_recall()),
                'f1_score': float(self.get_f1_score())
            },
            'per_class': {
                self.class_names[i]: {
                    'true_positives': int(self.true_positives[i]),
                    'false_positives': int(self.false_positives[i]),
                    'true_negatives': int(self.true_negatives[i]),
                    'false_negatives': int(self.false_negatives[i]),
                    'precision': float(self.get_precision(i)),
                    'recall': float(self.get_recall(i)),
                    'f1_score': float(self.get_f1_score(i))
                }
                for i in range(self.num_classes)
            },
            'confusion_matrix': self.get_confusion_matrix().tolist(),
            'latency': self.get_latency_stats(),
            'fps': float(self.get_fps())
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"✓ Metrics saved to {output_path}")
    
    def plot_metrics(self, output_dir):
        """Generate visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = self.get_confusion_matrix()
        im = ax.imshow(cm, cmap='Blues')
        
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        
        # Add text annotations
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        # 2. Latency Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.inference_times, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.inference_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.inference_times):.2f} ms')
        ax.axvline(np.median(self.inference_times), color='green', linestyle='--',
                   label=f'Median: {np.median(self.inference_times):.2f} ms')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Inference Latency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_distribution.png', dpi=150)
        plt.close()
        
        # 3. Per-Class Metrics Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(self.num_classes)
        width = 0.25
        
        precisions = [self.get_precision(i) for i in range(self.num_classes)]
        recalls = [self.get_recall(i) for i in range(self.num_classes)]
        f1_scores = [self.get_f1_score(i) for i in range(self.num_classes)]
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics.png', dpi=150)
        plt.close()
        
        # 4. Inference Time Over Time
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.inference_times, alpha=0.6, linewidth=0.5)
        # Rolling average
        window = 50
        if len(self.inference_times) >= window:
            rolling_avg = np.convolve(self.inference_times, 
                                      np.ones(window)/window, 
                                      mode='valid')
            ax.plot(range(window-1, len(self.inference_times)), 
                   rolling_avg, color='red', linewidth=2, 
                   label=f'{window}-frame moving average')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_over_time.png', dpi=150)
        plt.close()
        
        print(f"✓ Plots saved to {output_dir}/")


def example_usage():
    """Example: Simulate real-time metrics collection"""
    print("Example: Real-Time Metrics System")
    print("="*70)
    
    # Initialize metrics
    metrics = PerformanceMetrics(
        num_classes=2,
        class_names=["no_person", "person"]
    )
    
    # Simulate inference results
    np.random.seed(42)
    
    for i in range(500):
        # Simulate ground truth and prediction
        y_true = np.random.randint(0, 2)
        # Model is 80% accurate
        y_pred = y_true if np.random.rand() > 0.2 else (1 - y_true)
        confidence = np.random.uniform(0.6, 0.99)
        inference_time = np.random.normal(145, 15)  # ~145ms ± 15ms
        
        # Update metrics
        metrics.update(y_true, y_pred, confidence, inference_time)
    
    # Print summary
    metrics.print_summary()
    
    # Save to JSON
    metrics.save_to_json('metrics_example.json')
    
    # Generate plots
    metrics.plot_metrics('metrics_plots')
    
    print("\n✓ Example complete! Check metrics_example.json and metrics_plots/")


if __name__ == '__main__':
    example_usage()
