"""
Inspect the contents of the results NPZ files
"""

import numpy as np
import os

def inspect_npz_file(filepath):
    """Inspect the contents of an NPZ file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    try:
        # Load the NPZ file with pickle allowed
        data = np.load(filepath, allow_pickle=True)
        
        print(f"File size: {os.path.getsize(filepath)} bytes")
        print(f"Number of arrays: {len(data.files)}")
        print(f"Array names: {data.files}")
        
        # Display details for each array
        for name in data.files:
            array = data[name]
            print(f"\n--- Array: '{name}' ---")
            print(f"  Shape: {array.shape}")
            print(f"  Data type: {array.dtype}")
            
            # Handle different data types
            if name == 'L':
                # Likelihood values
                print(f"  Min value: {np.min(array):.6f}")
                print(f"  Max value: {np.max(array):.6f}")
                print(f"  Mean value: {np.mean(array):.6f}")
                print(f"  Values: {array}")
            elif name == 'params':
                # Parameters (likely a dictionary or object)
                print(f"  Content type: {type(array.item())}")
                if array.shape == ():
                    # Scalar array containing an object
                    params_obj = array.item()
                    if isinstance(params_obj, dict):
                        print(f"  Dictionary keys: {list(params_obj.keys())}")
                        for key, value in params_obj.items():
                            if isinstance(value, np.ndarray):
                                print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                                if value.size < 20:
                                    print(f"      Values: {value}")
                                else:
                                    print(f"      Sample: {value.flat[:10]}...")
                            else:
                                print(f"    {key}: {value}")
                    else:
                        print(f"  Content: {params_obj}")
                else:
                    print(f"  Content: {array}")
            else:
                # Other arrays
                if array.dtype.kind in ['f', 'i', 'u']:  # numeric types
                    print(f"  Min value: {np.min(array):.6f}")
                    print(f"  Max value: {np.max(array):.6f}")
                    print(f"  Mean value: {np.mean(array):.6f}")
                
                # Show sample values
                if array.ndim == 1:
                    print(f"  Sample values (first 10): {array[:10]}")
                elif array.ndim == 2:
                    print(f"  Sample values (first 3x3):")
                    print(f"    {array[:3, :3]}")
                elif array.ndim == 3:
                    print(f"  Sample values (first slice [0, :3, :3]):")
                    print(f"    {array[0, :3, :3]}")
                else:
                    print(f"  Sample values: {array.flat[:10]}")
        
        data.close()
        
    except Exception as e:
        print(f"Error loading file: {e}")

def main():
    """Main inspection function"""
    print("NPZ File Inspector")
    print("="*60)
    
    # Check current result files (after cleanup)
    files_to_inspect = [
        "results/test_real_data_topic2.npz",  # Real data, 15D features, 2 topics
        "results/test_real_data_topic3.npz",  # Real data, 15D features, 3 topics
        "results/test_synthetic_topic1.npz",  # Synthetic data, 1 topic
        "results/test_synthetic_topic2.npz",  # Synthetic data, 2 topics
    ]
    
    for filepath in files_to_inspect:
        inspect_npz_file(filepath)
    
    print(f"\n{'='*60}")
    print("Inspection completed!")

if __name__ == "__main__":
    main()
