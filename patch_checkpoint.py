import json
import os
import shutil
from safetensors.torch import load_file, save_file
from collections import defaultdict

# This is the definitive list of modules that vLLM v0.9.1 expects
# to be quantized in the Llama architecture.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def is_target_tensor(key: str) -> bool:
    """
    Checks if a tensor key belongs to a module that should be quantized.
    We will only rename the .weight of these specific modules.
    """
    return any(f".{module}." in key for module in TARGET_MODULES)

def patch_checkpoint(model_path: str, output_path: str):
    """
    Patches a sharded or single-file AWQ model checkpoint from llmcompressor
    to be compatible with vLLM v0.9.1.
    """
    print(f"Starting patch process for model at: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Source model path not found: {model_path}")

    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Deleting it for a clean run.")
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    print(f"Created clean output directory: {output_path}")

    # --- 1. Patch the Quantization Config ---
    quant_config_found = False
    for config_name in ["quant_config.json", "quantization_config.json"]:
        quant_config_path = os.path.join(model_path, config_name)
        if os.path.exists(quant_config_path):
            print(f"Found quantization config: {config_name}")
            with open(quant_config_path, 'r') as f:
                config = json.load(f)

            config["quant_method"] = "awq"
            config["version"] = "v2"

            output_config_path = os.path.join(output_path, "quant_config.json")
            with open(output_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Successfully patched and saved config to: {output_config_path}")
            quant_config_found = True
            break
    
    if not quant_config_found:
        print("WARNING: No quantization config file found. Continuing without it.")

    # --- 2. Patch the Model Tensors ---
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path):
        # --- Sharded Model ---
        print("Detected sharded model. Processing each shard...")
        with open(index_path, 'r') as f:
            index = json.load(f)
        shutil.copy(index_path, output_path)

        # === THIS IS THE CORRECTED LOGIC ===
        # Get the unique set of shard filenames from the index's values
        shard_filenames = sorted(list(set(index["weight_map"].values())))
        print(f"Found {len(shard_filenames)} unique shard files to process.")

        for shard_filename in shard_filenames:
            shard_path = os.path.join(model_path, shard_filename)
            if not os.path.exists(shard_path):
                print(f"  WARNING: Could not find shard file {shard_filename}, skipping.")
                continue

            print(f"  Processing shard: {shard_filename}...")
            
            tensors = load_file(shard_path)
            renamed_tensors = {}
            for key, value in tensors.items():
                if key.endswith(".weight") and is_target_tensor(key):
                    new_key = key.replace(".weight", ".qweight")
                    print(f"    Renaming {key} -> {new_key}")
                    renamed_tensors[new_key] = value
                else:
                    renamed_tensors[key] = value

            output_shard_path = os.path.join(output_path, shard_filename)
            save_file(renamed_tensors, output_shard_path)
            print(f"  Saved patched shard to: {output_shard_path}")
        # === END OF CORRECTED LOGIC ===

    else:
        # --- Single-File Model ---
        single_file_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single_file_path):
            raise FileNotFoundError("Could not find sharded index or single model.safetensors file.")
        
        print("Detected single-file model. Processing...")
        tensors = load_file(single_file_path)
        renamed_tensors = {}
        for key, value in tensors.items():
            if key.endswith(".weight") and is_target_tensor(key):
                new_key = key.replace(".weight", ".qweight")
                print(f"  Renaming {key} -> {new_key}")
                renamed_tensors[new_key] = value
            else:
                renamed_tensors[key] = value
        
        output_model_path = os.path.join(output_path, "model.safetensors")
        save_file(renamed_tensors, output_model_path)
        print(f"Saved patched model to: {output_model_path}")

    # --- 3. Copy All Other Necessary Files ---
    print("Copying remaining tokenizer and config files...")
    for filename in os.listdir(model_path):
        # We only need to copy files that are not part of the model weights/config
        # since we handled those already.
        if filename.endswith((".safetensors", ".json", ".py", ".md")):
            continue
        
        # A simple way to copy other essential files like tokenizer.model
        source_file = os.path.join(model_path, filename)
        if os.path.isfile(source_file):
             print(f"  Copying {filename}...")
             shutil.copy(source_file, output_path)

    # We also need to copy the tokenizer config and other essential json files
    # that weren't the quant_config.json
    for filename in os.listdir(model_path):
        if filename.endswith(".json") and "quant" not in filename:
            source_file = os.path.join(model_path, filename)
            if os.path.isfile(source_file):
                print(f"  Copying {filename}...")
                shutil.copy(source_file, output_path)


    print("Copying complete.")
    print("\nPatching process finished successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch an llmcompressor AWQ model for vLLM v0.9.1.")
    parser.add_argument("model_path", type=str, help="Path to the original quantized model directory.")
    parser.add_argument("output_path", type=str, help="Path to save the patched model directory.")
    args = parser.parse_args()
    patch_checkpoint(args.model_path, args.output_path)
