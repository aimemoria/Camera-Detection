"""
TinyML Person Detection - TensorFlow Lite Conversion & Quantization

This script:
1. Loads trained Keras model (.h5)
2. Converts to TensorFlow Lite format
3. Applies INT8 post-training quantization
4. Compares model sizes (float32 vs int8)
5. Validates converted model accuracy
6. Generates C header file for Arduino deployment
7. Ensures compatibility with TensorFlow Lite for Microcontrollers

INT8 Quantization Benefits:
- Reduces model size by ~4x (float32 → int8)
- Faster inference on microcontrollers
- Lower memory usage
- Hardware acceleration support (if available)

Typical size reduction: 200 KB → 50 KB
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import argparse


class TFLiteConverter:
    """Convert and quantize Keras model to TensorFlow Lite"""
    
    def __init__(self, model_path, output_dir='tflite_models'):
        """
        Args:
            model_path: Path to trained Keras model (.h5)
            output_dir: Directory to save TFLite models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model
        print("Loading Keras model...")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"✓ Model loaded from {self.model_path}")
        
        self.model.summary()
        
        # Model info
        self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
        self.input_dtype = self.model.input.dtype
    
    def representative_dataset_generator(self, dataset_dir, num_samples=100):
        """
        Generate representative dataset for quantization calibration
        
        This is CRITICAL for INT8 quantization. The converter needs to see
        actual data to determine optimal quantization parameters.
        
        Args:
            dataset_dir: Path to preprocessed dataset
            num_samples: Number of samples to use for calibration
        
        Yields:
            Single input sample for calibration
        """
        dataset_dir = Path(dataset_dir)
        
        # Collect sample files
        sample_files = []
        for cls in ['person', 'no_person']:
            train_dir = dataset_dir / 'train' / cls
            if train_dir.exists():
                npy_files = list(train_dir.glob('*.npy'))
                sample_files.extend(npy_files[:num_samples // 2])
        
        print(f"Using {len(sample_files)} samples for quantization calibration")
        
        # Generate samples
        for npy_file in sample_files:
            img = np.load(npy_file)
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            yield [img.astype(np.float32)]
    
    def convert_to_tflite_float32(self):
        """
        Convert to TFLite with float32 weights (no quantization)
        
        This is mainly for comparison purposes.
        """
        print("\n" + "="*70)
        print("CONVERTING TO TFLITE (FLOAT32)")
        print("="*70)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # No optimizations - keep float32
        converter.optimizations = []
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        output_path = self.output_dir / 'person_detector_float32.tflite'
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✓ Float32 model saved to {output_path}")
        print(f"  Size: {size_kb:.2f} KB ({len(tflite_model):,} bytes)")
        
        return output_path, len(tflite_model)
    
    def convert_to_tflite_int8(self, dataset_dir):
        """
        Convert to TFLite with INT8 quantization
        
        This is the PRIMARY format for Arduino deployment.
        
        Args:
            dataset_dir: Path to preprocessed dataset for representative data
        
        Returns:
            Path to saved model, model size in bytes
        """
        print("\n" + "="*70)
        print("CONVERTING TO TFLITE (INT8 QUANTIZED)")
        print("="*70)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Provide representative dataset for calibration
        converter.representative_dataset = lambda: self.representative_dataset_generator(
            dataset_dir
        )
        
        # Ensure INT8 for all operations (strict quantization)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Set input/output types to INT8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert
        print("Converting... (this may take a minute)")
        tflite_model = converter.convert()
        
        # Save
        output_path = self.output_dir / 'person_detector_int8.tflite'
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        print(f"✓ INT8 model saved to {output_path}")
        print(f"  Size: {size_kb:.2f} KB ({len(tflite_model):,} bytes)")
        
        # Check if within constraints
        if size_kb > 100:
            print(f"  ⚠️  WARNING: Model exceeds 100 KB target ({size_kb:.2f} KB)")
        else:
            print(f"  ✓ Model fits within 100 KB constraint")
        
        return output_path, len(tflite_model)
    
    def validate_tflite_model(self, tflite_path, dataset_dir, num_samples=100):
        """
        Validate TFLite model accuracy
        
        Args:
            tflite_path: Path to TFLite model
            dataset_dir: Path to preprocessed dataset
            num_samples: Number of test samples to validate
        
        Returns:
            Accuracy percentage
        """
        print("\n" + "="*70)
        print(f"VALIDATING MODEL: {tflite_path.name}")
        print("="*70)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details:")
        print(f"  Shape: {input_details[0]['shape']}")
        print(f"  Dtype: {input_details[0]['dtype']}")
        
        print(f"Output details:")
        print(f"  Shape: {output_details[0]['shape']}")
        print(f"  Dtype: {output_details[0]['dtype']}")
        
        # Load test samples
        dataset_dir = Path(dataset_dir)
        test_samples = []
        test_labels = []
        
        for label, cls in enumerate(['no_person', 'person']):
            test_dir = dataset_dir / 'test' / cls
            if test_dir.exists():
                npy_files = list(test_dir.glob('*.npy'))[:num_samples // 2]
                for npy_file in npy_files:
                    img = np.load(npy_file)
                    test_samples.append(img)
                    test_labels.append(label)
        
        print(f"\nTesting on {len(test_samples)} samples...")
        
        # Run inference
        correct = 0
        total = len(test_samples)
        
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']
        
        for i, (img, label) in enumerate(zip(test_samples, test_labels)):
            # Prepare input
            input_data = np.expand_dims(img, axis=0).astype(np.float32)
            
            # Quantize input if needed
            if input_details[0]['dtype'] == np.int8:
                input_data = input_data / input_scale + input_zero_point
                input_data = np.clip(input_data, -128, 127).astype(np.int8)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Dequantize output if needed
            if output_details[0]['dtype'] == np.int8:
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Get prediction
            pred_label = np.argmax(output_data[0])
            
            if pred_label == label:
                correct += 1
        
        accuracy = correct / total
        print(f"\nValidation Results:")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        
        return accuracy
    
    def generate_c_header(self, tflite_path, header_name='model_data.h'):
        """
        Generate C header file for Arduino deployment
        
        Converts .tflite binary to C array that can be compiled into Arduino sketch.
        
        Args:
            tflite_path: Path to TFLite model
            header_name: Name of output header file
        
        Returns:
            Path to generated header file
        """
        print("\n" + "="*70)
        print("GENERATING C HEADER FILE")
        print("="*70)
        
        # Read TFLite model
        with open(tflite_path, 'rb') as f:
            tflite_binary = f.read()
        
        # Generate C array
        c_array = ','.join([f'0x{b:02x}' for b in tflite_binary])
        
        # Generate header content
        header_content = f'''/*
 * TinyML Person Detection Model
 * 
 * Auto-generated from: {tflite_path.name}
 * Model size: {len(tflite_binary):,} bytes ({len(tflite_binary)/1024:.2f} KB)
 * Input shape: {self.input_shape}
 * 
 * Generated by TFLite converter for Arduino deployment
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Model size
const unsigned int model_data_len = {len(tflite_binary)};

// Model data
const unsigned char model_data[] = {{
  {c_array}
}};

#endif  // MODEL_DATA_H
'''
        
        # Save header file
        header_path = self.output_dir / header_name
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        print(f"✓ C header file saved to {header_path}")
        print(f"  Array name: model_data")
        print(f"  Array size: {len(tflite_binary):,} bytes")
        print(f"  Use this file in your Arduino sketch")
        
        return header_path
    
    def save_conversion_summary(self, float32_size, int8_size, int8_accuracy):
        """Save conversion summary"""
        summary = {
            'original_model': str(self.model_path),
            'input_shape': list(self.input_shape),
            'keras_parameters': int(self.model.count_params()),
            'float32_tflite_size_bytes': int(float32_size),
            'float32_tflite_size_kb': float(float32_size / 1024),
            'int8_tflite_size_bytes': int(int8_size),
            'int8_tflite_size_kb': float(int8_size / 1024),
            'size_reduction_ratio': float(float32_size / int8_size),
            'size_reduction_percent': float((1 - int8_size / float32_size) * 100),
            'int8_validation_accuracy': float(int8_accuracy),
            'within_100kb_constraint': int8_size <= 100 * 1024
        }
        
        summary_path = self.output_dir / 'conversion_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Conversion summary saved to {summary_path}")
        
        # Print summary table
        print("\n" + "="*70)
        print("CONVERSION SUMMARY")
        print("="*70)
        print(f"Original Keras model:      {self.model_path}")
        print(f"Total parameters:          {summary['keras_parameters']:,}")
        print(f"Float32 TFLite size:       {summary['float32_tflite_size_kb']:.2f} KB")
        print(f"INT8 TFLite size:          {summary['int8_tflite_size_kb']:.2f} KB")
        print(f"Size reduction:            {summary['size_reduction_ratio']:.2f}x ({summary['size_reduction_percent']:.1f}%)")
        print(f"INT8 accuracy:             {summary['int8_validation_accuracy']*100:.2f}%")
        print(f"Fits in 100 KB constraint: {'YES ✓' if summary['within_100kb_constraint'] else 'NO ✗'}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Keras model to TensorFlow Lite with INT8 quantization'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained Keras model (.h5 file)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to preprocessed dataset (for representative data)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tflite_models',
        help='Output directory for TFLite models (default: tflite_models)'
    )
    parser.add_argument(
        '--skip_float32',
        action='store_true',
        help='Skip float32 conversion (only generate INT8)'
    )
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Skip model validation'
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = TFLiteConverter(
        model_path=args.model,
        output_dir=args.output_dir
    )
    
    # Convert to float32 (optional)
    if not args.skip_float32:
        float32_path, float32_size = converter.convert_to_tflite_float32()
    else:
        float32_size = None
    
    # Convert to INT8 (main target)
    int8_path, int8_size = converter.convert_to_tflite_int8(args.dataset_dir)
    
    # Validate INT8 model
    if not args.skip_validation:
        int8_accuracy = converter.validate_tflite_model(
            int8_path, 
            args.dataset_dir,
            num_samples=200
        )
    else:
        int8_accuracy = None
    
    # Generate C header file for Arduino
    header_path = converter.generate_c_header(int8_path)
    
    # Save summary
    if float32_size and int8_accuracy:
        converter.save_conversion_summary(float32_size, int8_size, int8_accuracy)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"INT8 model:   {int8_path}")
    print(f"C header:     {header_path}")
    print(f"Output dir:   {args.output_dir}")
    print("\nNext step: Deploy model_data.h to Arduino")


if __name__ == '__main__':
    main()
