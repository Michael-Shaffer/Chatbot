import json
import os
import argparse
import shutil
from safetensors.torch import load_file, save_file
import torch

# This mapping is based on common conventions. The keys are potential suffixes from
# llmcompressor/compressed-tensors, and values are the expected suffixes for vLLM's AWQ loader.
TENSOR_NAME_MAP = {
    "weight": "qweight",
    "scales": "scales",
    "zeros": "qzeros",
}

def patch_sharded_checkpoint(model_path, output_path):
    """Patches a sharded model checkpoint."""
    print("--- Detected sharded model. Patching shards... ---")
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    total_renamed_count = 0
    shard_filenames = set(index["weight_map"].values())

    for shard_filename in shard_filenames:
        print(f"  Processing shard: {shard_filename}")
        shard_path = os.path.join(model_path, shard_filename)
        
        state_dict = load_file(shard_path)
        new_state_dict = {}
        renamed_in_shard = 0

        for key, tensor in state_dict.items():
            new_key = key
            # *** FIX: Do not rename the lm_head layer ***
            if 'lm_head' in key:
                new_state_dict[new_key] = tensor
                continue

            for old_suffix, new_suffix in TENSOR_NAME_MAP.items():
                if key.endswith(f".{old_suffix}"):
                    new_key = key[:-len(old_suffix)] + new_suffix
                    break
            
            if new_key!= key:
                print(f"    Renaming tensor: '{key}' -> '{new_key}'")
                renamed_in_shard += 1
            
            new_state_dict[new_key] = tensor
        
        print(f"    Renamed {renamed_in_shard} tensors in this shard.")
        total_renamed_count += renamed_in_shard

        new_shard_path = os.path.join(output_path, shard_filename)
        save_file(new_state_dict, new_shard_path)

    print(f"Total tensors renamed across all shards: {total_renamed_count}")
    
    shutil.copy(index_path, os.path.join(output_path, "model.safetensors.index.json"))
    print("Copied model index file.")

def patch_single_file_checkpoint(model_path, output_path):
    """Patches a single-file model checkpoint."""
    print("--- Detected single-file model. Patching... ---")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    
    state_dict = load_file(safetensors_path)
    new_state_dict = {}
    renamed_count = 0

    for key, tensor in state_dict.items():
        new_key = key
        # *** FIX: Do not rename the lm_head layer ***
        if 'lm_head' in key:
            new_state_dict[new_key] = tensor
            continue

        for old_suffix, new_suffix in TENSOR_NAME_MAP.items():
            if key.endswith(f".{old_suffix}"):
                new_key = key[:-len(old_suffix)] + new_suffix
                break
        
        if new_key!= key:
            print(f"  Renaming tensor: '{key}' -> '{new_key}'")
            renamed_count += 1
            
        new_state_dict[new_key] = tensor

    print(f"Total tensors renamed: {renamed_count}")
    
    new_safetensors_path = os.path.join(output_path, "model.safetensors")
    save_file(new_state_dict, new_safetensors_path)
    print(f"Patched model.safetensors saved to: {new_safetensors_path}")


def patch_checkpoint(model_path, output_path):
    """
    Patches an AWQ model checkpoint (single-file or sharded) to be compatible with vLLM.
    """
    print(f"Starting patch process for model at: {model_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_file_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(index_path):
        patch_sharded_checkpoint(model_path, output_path)
    elif os.path.exists(single_file_path):
        patch_single_file_checkpoint(model_path, output_path)
    else:
        raise FileNotFoundError(
            f"Could not find 'model.safetensors' or 'model.safetensors.index.json' in {model_path}"
        )

    config_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(config_path):
        config_path_alt = os.path.join(model_path, "quant_config.json")
        if os.path.exists(config_path_alt):
            config_path = config_path_alt
            print("Note: Using 'quant_config.json' as quantization configuration file.")
        else:
            raise FileNotFoundError(f"Could not find 'quantization_config.json' or 'quant_config.json' in {model_path}")

    with open(config_path, "r") as f:
        quant_config = json.load(f)

    quant_config.setdefault("quant_method", "awq")
    quant_config.setdefault("version", "GEMM")

    new_config_path = os.path.join(output_path, "quantization_config.json")
    with open(new_config_path, "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"Normalized quantization_config.json saved to: {new_config_path}")

    print("Copying remaining model files (tokenizer, etc.)...")
    copied_files = 0
    for filename in os.listdir(model_path):
        if "safetensors" in filename or "quant_config" in filename:
            continue
        
        source_file = os.path.join(model_path, filename)
        dest_file = os.path.join(output_path, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, dest_file)
            copied_files += 1
    print(f"Copied {copied_files} remaining model files.")
    print("\nPatching complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patch an llmcompressor AWQ model (sharded or single-file) to be compatible with vLLM."
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
    args = parser.parse_args()
    patch_checkpoint(args.model_path, args.output_path)
