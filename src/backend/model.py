# llm.py
# Contains functions for loading and interacting with the local LLM (Llama.cpp).
# Last updated: 2025-05-22

import os
from llama_cpp import Llama

# Global variable to cache the LLM instance
_llm_instance_cache = None

def get_llm_instance(model_path, n_gpu_layers=-1, n_ctx=4096):
    """
    Loads or returns a cached Llama.cpp model instance.
    This runs the model directly in the Python process.
    """
    global _llm_instance_cache
    if _llm_instance_cache is None:
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model file not found at {model_path}. Please provide a valid path.")
        
        print(f"Loading Llama.cpp model from: {model_path} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")
        try:
            _llm_instance_cache = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False 
            )
            print("Llama.cpp model loaded successfully.")
        except Exception as e:
            print(f"Error loading Llama.cpp model: {e}")
            print("Ensure you have the correct GGUF model file and llama-cpp-python installed with appropriate hardware support (e.g., BLAS/AVX2 for CPU, CUDA/Metal for GPU).")
            raise 
    return _llm_instance_cache

def get_llm_response_self_contained(prompt_text, llm_model_path_for_init, max_new_tokens=500, current_n_ctx=4096, current_n_gpu_layers=-1):
    """
    Gets a response from the self-contained Llama.cpp model using the cached instance.
    """
    try:
        # Ensure the LLM instance is loaded using the provided parameters
        llm = get_llm_instance(model_path=llm_model_path_for_init, n_ctx=current_n_ctx, n_gpu_layers=current_n_gpu_layers)
        
        # Llama 3.1 Instruct prompt format
        # Ensure your prompt_text ALREADY CONTAINS the context and question as per your RAG logic.
        # This function just handles the final LLM call formatting.
        system_message = "" # You can add a system message if desired, e.g., for Llama 3.1: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
        
        # The prompt_text from RAG should be the user's part of the conversation
        full_prompt_for_llm = f"{system_message}<|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # print(f"\nSending prompt to self-contained Llama.cpp model...")
        # print(f"--- PROMPT START ---\n{full_prompt_for_llm}\n--- PROMPT END ---") # For debugging prompt structure

        output = llm(
            full_prompt_for_llm,
            max_tokens=max_new_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"], # Common stop tokens for Llama 3
            echo=False 
        )
        response_text = output['choices'][0]['text'].strip()
        # print("LLM response received.")
        return response_text
    except Exception as e:
        print(f"Error during Llama.cpp inference: {e}")
        return "Sorry, I encountered an error trying to generate a response with the local LLM."
