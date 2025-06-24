#!/usr/bin/env python3
"""
Debug version to see exactly what vLLM is returning
"""

import os
import time
import json
import logging

# Set environment variables
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache" 
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)

from flask import Flask, request, render_template_string, Response
from vllm import LLM, SamplingParams

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize vLLM (only load once)
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Prevent double loading in Flask debug mode
if 'llm' not in globals():
    logger.info("Loading vLLM model...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.5
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"]
    )

    logger.info("Model loaded successfully!")
else:
    logger.info("Model already loaded, skipping...")

def format_chat_prompt(user_message: str) -> str:
    """Format prompt for Llama 3.1 chat template"""
    system_prompt = "You are a helpful assistant."
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def debug_generate_response(user_message: str) -> str:
    """Generate response with detailed debugging"""
    try:
        logger.info(f"=== DEBUGGING GENERATION FOR: {user_message} ===")
        
        # Format the prompt
        formatted_prompt = format_chat_prompt(user_message)
        logger.debug(f"Formatted prompt: {formatted_prompt[:200]}...")
        
        # Generate response
        logger.info("Calling llm.generate()...")
        outputs = llm.generate([formatted_prompt], sampling_params)
        
        # Debug the outputs structure
        logger.info(f"Type of outputs: {type(outputs)}")
        logger.info(f"Length of outputs: {len(outputs)}")
        
        if outputs and len(outputs) > 0:
            output = outputs[0]
            logger.info(f"Type of output[0]: {type(output)}")
            logger.info(f"Output[0] attributes: {dir(output)}")
            
            if hasattr(output, 'outputs'):
                logger.info(f"output.outputs type: {type(output.outputs)}")
                logger.info(f"output.outputs length: {len(output.outputs)}")
                
                if len(output.outputs) > 0:
                    completion = output.outputs[0]
                    logger.info(f"Type of completion: {type(completion)}")
                    logger.info(f"Completion attributes: {dir(completion)}")
                    
                    if hasattr(completion, 'text'):
                        response_text = completion.text
                        logger.info(f"Raw response text: '{response_text}'")
                        logger.info(f"Response text length: {len(response_text)}")
                        
                        stripped_response = response_text.strip()
                        logger.info(f"Stripped response: '{stripped_response}'")
                        return stripped_response
                    else:
                        logger.error("completion object has no 'text' attribute")
                        return f"DEBUG: completion has no text. Attributes: {dir(completion)}"
                else:
                    logger.error("output.outputs is empty")
                    return "DEBUG: output.outputs is empty"
            else:
                logger.error("output has no 'outputs' attribute")
                return f"DEBUG: output has no outputs. Attributes: {dir(output)}"
        else:
            logger.error("outputs is empty or None")
            return "DEBUG: No outputs generated"
            
    except Exception as e:
        logger.error(f"Error in generation: {e}", exc_info=True)
        return f"ERROR: {str(e)}"

# Flask app
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug vLLM Response</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .chat-container { 
            background: white;
            border: 1px solid #ddd; 
            height: 500px; 
            overflow-y: scroll; 
            padding: 20px; 
            margin: 20px 0;
            border-radius: 8px;
        }
        .message { 
            margin: 15px 0; 
            padding: 12px; 
            border-radius: 8px; 
            max-width: 80%;
        }
        .user { 
            background-color: #007bff; 
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .assistant { 
            background-color: #f8f9fa; 
            border: 1px solid #dee2e6;
            white-space: pre-wrap;
        }
        .debug {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        input[type="text"] { 
            flex: 1;
            padding: 12px; 
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button { 
            padding: 12px 24px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #6c757d; cursor: not-allowed; }
        h1 { text-align: center; color: #333; }
    </style>
</head>
<body>
    <h1>🔍 Debug vLLM Response</h1>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Type 'hello' to test..." 
               onkeypress="if(event.key==='Enter') sendMessage()">
        <button id="sendButton" onclick="sendMessage()">Send</button>
    </div>

    <script>
        let isGenerating = false;

        function sendMessage() {
            if (isGenerating) return;
            
            const input = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            isGenerating = true;
            sendButton.disabled = true;
            sendButton.textContent = 'Debugging...';
            
            fetch('/debug_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(response => {
                console.log('Response status:', response.status);
                console.log('Response headers:', response.headers);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    console.error('Response is not JSON:', contentType);
                    return response.text().then(text => {
                        console.error('Response text:', text);
                        throw new Error('Server returned non-JSON response');
                    });
                }
                
                return response.json();
            })
            .then(data => {
                if (data.response) {
                    addMessage(data.response, 'assistant');
                } else {
                    addMessage('No response in data: ' + JSON.stringify(data), 'debug');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Fetch error: ' + error, 'debug');
            })
            .finally(() => {
                isGenerating = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                input.focus();
            });
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }
        
        window.onload = function() {
            addMessage('Debug mode active. Check console logs for detailed vLLM output info.', 'debug');
            document.getElementById('messageInput').focus();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/debug_chat', methods=['POST'])
def debug_chat():
    """Debug endpoint to see exactly what's happening"""
    try:
        # Get the request data safely
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data'}), 400
            
        user_message = data.get('message', '')
        
        if not user_message:
            logger.error("No message in request")
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"\n{'='*50}")
        logger.info(f"DEBUG CHAT REQUEST: {user_message}")
        logger.info(f"{'='*50}")
        
        # Check if model is loaded
        if 'llm' not in globals() or llm is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Generate response with debugging
        response = debug_generate_response(user_message)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"FINAL RESPONSE: '{response}'")
        logger.info(f"RESPONSE LENGTH: {len(response)}")
        logger.info(f"{'='*50}\n")
        
        # Ensure we return valid JSON
        return jsonify({'response': response, 'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error in debug_chat: {e}", exc_info=True)
        # Return JSON error instead of letting Flask return HTML
        return jsonify({'error': f'Debug chat error: {str(e)}', 'status': 'error'}), 500

@app.route('/health')
def health():
    return {"status": "healthy", "mode": "debug"}

if __name__ == '__main__':
    print("\\n" + "="*50)
    print("🔍 DEBUG vLLM Response Extraction")
    print("📍 URL: http://localhost:8000")
    print("📊 Check console logs for detailed debugging info")
    print("="*50 + "\\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)  # Changed to debug=False
