import json
import os
import argparse
from safetensors.torch import load_file, save_file
import torch

# This mapping is based on common conventions. The keys are potential suffixes from
# llmcompressor/compressed-tensors, and values are the expected suffixes for vLLM's AWQ loader.
# This may need to be adjusted based on inspecting the actual tensor names in your model.
TENSOR_NAME_MAP = {
    "weight": "qweight",
    "scales": "scales",
    "zeros": "qzeros",
}

def inspect_model(model_path):
    """Prints the tensor keys and quantization config of a model."""
    print("--- Inspecting Model ---")
    
    # Load and print quantization config
    config_path = os.path.join(model_path, "quantization_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print("\nquantization_config.json:")
        print(json.dumps(config, indent=2))
    else:
        print("\nquantization_config.json not found.")

    # Load and print tensor keys from safetensors
    st_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(st_path):
        print(f"\nmodel.safetensors not found at {st_path}")
        return
        
    state_dict = load_file(st_path)
    print(f"\nTensor keys from {st_path}:")
    for key in state_dict.keys():
        print(key)
    print("--- End of Inspection ---\n")


def patch_checkpoint(model_path, output_path):
    """
    Patches an AWQ model checkpoint by renaming tensors and normalizing the config
    to be compatible with vLLM's AWQ loader.
    """
    print(f"Starting patch process for model at: {model_path}")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # --- Step 1: Load and patch the state dictionary ---
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Could not find model.safetensors in {model_path}")
        
    state_dict = load_file(safetensors_path)
    new_state_dict = {}
    renamed_count = 0

    for key, tensor in state_dict.items():
        new_key = key
        # Check if the key ends with one of the target suffixes
        for old_suffix, new_suffix in TENSOR_NAME_MAP.items():
            if key.endswith(f".{old_suffix}"):
                new_key = key[:-len(old_suffix)] + new_suffix
                break
        
        if new_key!= key:
            print(f"  Renaming tensor: '{key}' -> '{new_key}'")
            renamed_count += 1
            
        new_state_dict[new_key] = tensor

    print(f"Total tensors renamed: {renamed_count}")

    # --- Step 2: Load, normalize, and save the quantization config ---
    config_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find quantization_config.json in {model_path}")

    with open(config_path, "r") as f:
        quant_config = json.load(f)

    # Ensure necessary fields for vLLM are present.
    # vLLM's AWQ loader infers from these values.
    # The 'version' key is particularly important for some loaders.
    quant_config.setdefault("quant_method", "awq")
    quant_config.setdefault("version", "GEMM") # Common default for AutoAWQ models

    new_config_path = os.path.join(output_path, "quantization_config.json")
    with open(new_config_path, "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"Normalized quantization_config.json saved to: {new_config_path}")

    # --- Step 3: Save the patched state dictionary ---
    new_safetensors_path = os.path.join(output_path, "model.safetensors")
    save_file(new_state_dict, new_safetensors_path)
    print(f"Patched model.safetensors saved to: {new_safetensors_path}")

    # --- Step 4: Copy over any other necessary files ---
    for filename in os.listdir(model_path):
        if filename not in ["model.safetensors", "quantization_config.json"]:
            source_file = os.path.join(model_path, filename)
            dest_file = os.path.join(output_path, filename)
            if os.path.isfile(source_file):
                import shutil
                shutil.copy(source_file, dest_file)
    print("Copied remaining model files.")
    print("\nPatching complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patch an llmcompressor AWQ model to be compatible with vLLM."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the original AWQ model files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the directory where the patched model will be saved."
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect the model and print its keys and config without patching."
    )

    args = parser.parse_args()

    if args.inspect_only:
        inspect_model(args.model_path)
    else:
        patch_checkpoint(args.model_path, args.output_path)
