import numpy as np
import pandas as pd
from pathlib import Path

def main():
    # Define the base data directory relative to this script
    # Script is in src/, data is in data/ (project root/data)
    base_path = Path(__file__).resolve().parent.parent / 'data'
    
    if not base_path.exists():
        print(f"Error: Data directory not found at {base_path}")
        print("Please ensure you are running this script from the project root or that the data directory exists.")
        return

    # Iterate through function_1 to function_8
    for i in range(1, 9):
        func_name = f"function_{i}"
        func_dir = base_path / func_name
        
        if not func_dir.exists():
            print(f"Warning: Directory {func_dir} does not exist. Skipping.")
            continue
            
        inputs_path = func_dir / "initial_inputs.npy"
        outputs_path = func_dir / "initial_outputs.npy"
        
        if not inputs_path.exists() or not outputs_path.exists():
            print(f"Warning: .npy files not found in {func_dir}. Skipping.")
            continue
            
        try:
            # Load .npy files
            inputs = np.load(inputs_path)
            outputs = np.load(outputs_path)
            
            # Check shapes
            if inputs.ndim != 2:
                print(f"Error: {inputs_path} should be 2D. Got shape {inputs.shape}. Skipping.")
                continue
                
            N, D = inputs.shape
            
            # Handle outputs shape (N,) or (N, 1)
            if outputs.ndim > 1:
                outputs = outputs.flatten()
            
            if outputs.shape[0] != N:
                print(f"Error: Mismatch in number of samples. Inputs: {N}, Outputs: {outputs.shape[0]}. Skipping.")
                continue
            
            # Create column names for inputs: x0, x1, ...
            input_cols = [f"x{d}" for d in range(D)]
            
            # Create DataFrame
            df = pd.DataFrame(inputs, columns=input_cols)
            df['y'] = outputs
            df['source'] = 'initial'
            
            # Save to CSV
            csv_path = func_dir / "samples.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"Success: Created {csv_path.name} for {func_name} with shape {df.shape}")
            
        except Exception as e:
            print(f"Error processing {func_name}: {e}")

if __name__ == "__main__":
    main()
