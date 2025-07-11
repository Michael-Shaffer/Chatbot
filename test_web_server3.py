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

from flask import Flask, request, render_template_string, Response, jsonify
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
            
            # Print ALL attributes to see what's available
            output_attrs = [attr for attr in dir(output) if not attr.startswith('_')]
            logger.info(f"Output attributes: {output_attrs}")
            
            if hasattr(output, 'outputs'):
                logger.info(f"output.outputs type: {type(output.outputs)}")
                logger.info(f"output.outputs length: {len(output.outputs)}")
                
                if len(output.outputs) > 0:
                    completion = output.outputs[0]
                    logger.info(f"Type of completion: {type(completion)}")
                    
                    # Print ALL completion attributes
                    completion_attrs = [attr for attr in dir(completion) if not attr.startswith('_')]
                    logger.info(f"Completion attributes: {completion_attrs}")
                    
                    # Try different possible text attributes
                    for attr in ['text', 'generated_text', 'output', 'content']:
                        if hasattr(completion, attr):
                            value = getattr(completion, attr)
                            logger.info(f"completion.{attr}: '{value}' (type: {type(value)})")
                    
                    # Return whatever text we can find
                    if hasattr(completion, 'text'):
                        response_text = completion.text.strip()
                        logger.info(f"Using completion.text: '{response_text}'")
                        return response_text if response_text else "Empty response from model"
                    else:
                        # Try to find any text-like attribute
                        for attr in completion_attrs:
                            if 'text' in attr.lower():
                                value = getattr(completion, attr)
                                if isinstance(value, str) and value.strip():
                                    logger.info(f"Using {attr}: '{value}'")
                                    return value.strip()
                        
                        return f"DEBUG: No text found. Completion has: {completion_attrs}"
                else:
                    return "DEBUG: output.outputs is empty"
            else:
                return f"DEBUG: output has no outputs. Available: {output_attrs}"
        else:
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
            sendButton.textContent = 'Generating...';
            
            // Create assistant message div for streaming
            const currentAssistantDiv = addMessage('', 'assistant');
            
            // Use fetch with streaming instead of EventSource (more reliable)
            fetch('/stream?message=' + encodeURIComponent(message))
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    function readStream() {
                        return reader.read().then(({ done, value }) => {
                            if (done) {
                                console.log('Stream finished');
                                isGenerating = false;
                                sendButton.disabled = false;
                                sendButton.textContent = 'Send';
                                input.focus();
                                return;
                            }
                            
                            // Decode the chunk
                            const chunk = decoder.decode(value, { stream: true });
                            console.log('Received chunk:', chunk);
                            
                            const lines = chunk.split('\\n');
                            console.log('Split into lines:', lines);
                            
                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    console.log('Processing data line:', line);
                                    try {
                                        const jsonData = line.slice(6);
                                        console.log('JSON data:', jsonData);
                                        const data = JSON.parse(jsonData);
                                        console.log('Parsed data:', data);
                                        
                                        if (data.token) {
                                            console.log('Adding token to div:', data.token);
                                            currentAssistantDiv.textContent += data.token;
                                            console.log('Current div text:', currentAssistantDiv.textContent);
                                            scrollToBottom();
                                        } else if (data.done) {
                                            console.log('Received done signal');
                                            isGenerating = false;
                                            sendButton.disabled = false;
                                            sendButton.textContent = 'Send';
                                            input.focus();
                                            return;
                                        }
                                    } catch (e) {
                                        console.error('JSON parse error:', e, 'Line:', line);
                                    }
                                }
                            }
                            
                            // Continue reading
                            return readStream();
                        });
                    }
                    
                    return readStream();
                })
                .catch(error => {
                    console.error('Fetch streaming error:', error);
                    
                    if (currentAssistantDiv && currentAssistantDiv.textContent === '') {
                        currentAssistantDiv.textContent = 'Error: ' + error.message;
                    }
                    
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

@app.route('/stream')
def stream():
    """Stream the AI response using Server-Sent Events"""
    user_message = request.args.get('message', '')
    
    if not user_message:
        return Response("data: " + json.dumps({"error": "No message provided"}) + "\n\n", 
                       mimetype='text/plain')
    
    def generate():
        try:
            logger.info(f"STREAMING: Starting for message: {user_message}")
            
            # Generate the full response first (since vLLM doesn't stream natively)
            response = debug_generate_response(user_message)
            
            logger.info(f"STREAMING: Got response: '{response}'")
            logger.info(f"STREAMING: Response length: {len(response)}")
            logger.info(f"STREAMING: Response type: {type(response)}")
            
            if not response or response.strip() == "":
                logger.error("STREAMING: Response is empty!")
                yield f"data: {json.dumps({'token': 'Error: Empty response from model'})}\\n\\n"
                yield f"data: {json.dumps({'done': True})}\\n\\n"
                return
            
            # Split response into words and stream them
            words = response.split()
            logger.info(f"STREAMING: Split into {len(words)} words: {words}")
            
            for i, word in enumerate(words):
                if i == 0:
                    token = word
                else:
                    token = " " + word
                
                logger.info(f"STREAMING: Sending token {i}: '{token}'")
                
                # Send each word with a small delay for streaming effect
                data_line = f"data: {json.dumps({'token': token})}\\n\\n"
                logger.info(f"STREAMING: Yielding: {repr(data_line)}")
                yield data_line
                time.sleep(0.05)  # Small delay between words
            
            # Signal completion
            logger.info("STREAMING: Sending done signal")
            done_line = f"data: {json.dumps({'done': True})}\\n\\n"
            logger.info(f"STREAMING: Done line: {repr(done_line)}")
            yield done_line
            
        except Exception as e:
            logger.error(f"STREAMING: Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'token': f'Streaming Error: {str(e)}'})}\\n\\n"
            yield f"data: {json.dumps({'done': True})}\\n\\n"
    
    return Response(generate(), mimetype='text/plain')

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
