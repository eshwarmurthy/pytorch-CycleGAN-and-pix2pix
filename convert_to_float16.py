import onnx
from onnxconverter_common import float16, auto_convert_mixed_precision
import glob
import os
from pathlib import Path
import numpy as np


def convert_models_to_fp16(source_dir, target_dir):
    """
    Loads all FP32 ONNX models from a source directory, converts them to
    FP16 (Float16), and saves them to a target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    fp32_model_paths = glob.glob(f"{source_dir}/*.onnx")
    
    if not fp32_model_paths:
        print(f"Error: No .onnx models found in '{source_dir}'")
        return
        
    print(f"Found {len(fp32_model_paths)} models to convert to FP16.")
    
    for model_path in fp32_model_paths:
        model_name = Path(model_path).name
        fp16_model_path = os.path.join(target_dir, model_name)
        
        print(f"Converting {model_name} to FP16...")
        
        try:
            model = onnx.load(model_path)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, fp16_model_path)
            print(f"✓ Saved FP16 model to {fp16_model_path}\n")
            
        except Exception as e:
            print(f"✗ Failed to convert {model_name}: {e}\n")


def create_sample_input_data(model):
    """
    Create sample input data for the model based on its input specifications.
    """
    feed_dict = {}
    
    for input_tensor in model.graph.input:
        input_name = input_tensor.name
        input_shape = []
        
        # Extract shape from tensor type
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                input_shape.append(dim.dim_value)
            elif dim.dim_param:
                # Handle dynamic dimensions - use reasonable defaults
                if 'batch' in dim.dim_param.lower():
                    input_shape.append(1)  # batch size = 1
                else:
                    input_shape.append(256)  # default for other dynamic dims
            else:
                input_shape.append(1)  # fallback
        
        # Get data type
        data_type = input_tensor.type.tensor_type.elem_type
        
        # Create random input data based on type
        if data_type == onnx.TensorProto.FLOAT:
            # For image models, use values in range [0, 1] or [-1, 1]
            if len(input_shape) == 4 and input_shape[1] in [1, 3]:  # likely image: (batch, channels, height, width)
                sample_data = np.random.uniform(-1, 1, input_shape).astype(np.float32)
            else:
                sample_data = np.random.randn(*input_shape).astype(np.float32)
        elif data_type == onnx.TensorProto.INT64:
            sample_data = np.random.randint(0, 10, input_shape).astype(np.int64)
        elif data_type == onnx.TensorProto.INT32:
            sample_data = np.random.randint(0, 10, input_shape).astype(np.int32)
        else:
            # Default to float32
            sample_data = np.random.randn(*input_shape).astype(np.float32)
        
        feed_dict[input_name] = sample_data
        print(f"  Created sample input '{input_name}': shape {input_shape}, dtype {sample_data.dtype}")
    
    return feed_dict


def convert_models_to_mixed_precision(source_dir, target_dir, rtol=0.01, atol=0.001):
    """
    Loads all FP32 ONNX models from a source directory, converts them using
    auto mixed precision, and saves them to a target directory.
    
    Args:
        source_dir: Directory containing FP32 ONNX models
        target_dir: Directory to save mixed precision models
        rtol: Relative tolerance for mixed precision conversion
        atol: Absolute tolerance for mixed precision conversion
    """
    os.makedirs(target_dir, exist_ok=True)
    fp32_model_paths = glob.glob(f"{source_dir}/*.onnx")
    
    if not fp32_model_paths:
        print(f"Error: No .onnx models found in '{source_dir}'")
        return
        
    print(f"Found {len(fp32_model_paths)} models to convert to mixed precision.")
    
    for model_path in fp32_model_paths:
        model_name = Path(model_path).name
        mixed_precision_model_path = os.path.join(target_dir, model_name)
        
        print(f"Converting {model_name} to mixed precision...")
        
        try:
            model = onnx.load(model_path)
            
            # Create sample input data for the model
            print("  Generating sample input data...")
            feed_dict = create_sample_input_data(model)
            
            # Auto mixed precision conversion with tolerance settings
            model_mixed = auto_convert_mixed_precision(
                model, 
                feed_dict,
                rtol=rtol, 
                atol=atol,
                keep_io_types=True  # Keep input/output types as FP32 for compatibility
            )
            
            onnx.save(model_mixed, mixed_precision_model_path)
            print(f"✓ Saved mixed precision model to {mixed_precision_model_path}")
            print(f"  (rtol={rtol}, atol={atol})\n")
            
        except Exception as e:
            print(f"✗ Failed to convert {model_name}: {e}\n")


def convert_models_batch(source_dir, conversion_type="both", rtol=0.01, atol=0.001):
    """
    Batch convert models using the specified conversion type.
    
    Args:
        source_dir: Directory containing FP32 ONNX models
        conversion_type: "fp16", "mixed", or "both"
        rtol: Relative tolerance for mixed precision (only used if conversion_type includes mixed)
        atol: Absolute tolerance for mixed precision (only used if conversion_type includes mixed)
    """
    if conversion_type not in ["fp16", "mixed", "both"]:
        raise ValueError("conversion_type must be 'fp16', 'mixed', or 'both'")
    
    if conversion_type in ["fp16", "both"]:
        fp16_dir = os.path.join(source_dir, 'fp16_models')
        print("=== Converting to FP16 ===")
        convert_models_to_fp16(source_dir, fp16_dir)
    
    if conversion_type in ["mixed", "both"]:
        mixed_precision_dir = os.path.join(source_dir, 'mixed_precision_models')
        print("=== Converting to Mixed Precision ===")
        convert_models_to_mixed_precision(source_dir, mixed_precision_dir, rtol, atol)


def analyze_model_size_reduction(source_dir):
    """
    Analyze and compare file sizes between original, FP16, and mixed precision models.
    """
    original_models = glob.glob(f"{source_dir}/*.onnx")
    fp16_dir = os.path.join(source_dir, 'fp16_models')
    mixed_dir = os.path.join(source_dir, 'mixed_precision_models')
    
    if not original_models:
        print("No original models found for analysis.")
        return
    
    print("\n=== Model Size Analysis ===")
    print(f"{'Model Name':<30} {'Original (MB)':<15} {'FP16 (MB)':<12} {'Mixed (MB)':<12} {'FP16 Reduction':<15} {'Mixed Reduction':<15}")
    print("-" * 110)
    
    for model_path in original_models:
        model_name = Path(model_path).name
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        fp16_path = os.path.join(fp16_dir, model_name)
        mixed_path = os.path.join(mixed_dir, model_name)
        
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024) if os.path.exists(fp16_path) else 0
        mixed_size = os.path.getsize(mixed_path) / (1024 * 1024) if os.path.exists(mixed_path) else 0
        
        fp16_reduction = f"{((original_size - fp16_size) / original_size * 100):.1f}%" if fp16_size > 0 else "N/A"
        mixed_reduction = f"{((original_size - mixed_size) / original_size * 100):.1f}%" if mixed_size > 0 else "N/A"
        
        print(f"{model_name:<30} {original_size:<15.2f} {fp16_size:<12.2f} {mixed_size:<12.2f} {fp16_reduction:<15} {mixed_reduction:<15}")


if __name__ == "__main__":
    source_model_dir = '/imgarc/nila/data/Deblur_Defocus/Models'
    
    # Convert models using both methods
    convert_models_batch(source_model_dir, conversion_type="both", rtol=0.01, atol=0.001)
    
    # Analyze size reductions
    analyze_model_size_reduction(source_model_dir)
    
    # Alternative: Convert only to mixed precision with custom tolerances
    # convert_models_batch(source_model_dir, conversion_type="mixed", rtol=0.005, atol=0.0005)