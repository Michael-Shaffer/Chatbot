#!/usr/bin/env python3
"""
Python 3.9 Compatible vLLM Chatbot
Uses older vLLM API patterns to avoid Python 3.10+ features
"""

import os
import time
import json
import logging

# Disable vLLM tracking
os.environ["VLLM_DO_NOT_TRACK"] = "1"

from flask import Flask, request, render_template_string, Response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize vLLM model with Python 3.9 compatible settings
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Update this path

logger.info("Loading vLLM model with Python 3.9 compatibility...")

try:
    from vllm import LLM, SamplingParams
    
    # Use more conservative settings to avoid newer vLLM features
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        dtype="float16",
        max_model_len=2048,  # Smaller to be safer
        gpu_memory_utilization=0.5,  # More conservative
        trust_remote_code=True,
        disable_custom_all_reduce=True,  # Disable newer features
        enforce_eager=True,  # Use eager execution (older, more stable)
        # disable_sliding_window=True,  # Disable if this option exists
    )
    
    # Simple sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,  # Shorter responses to be safer
        stop=["</s>", "<|im_end|>", "<|endoftext|>"]
    )
    
    MODEL_LOADED = True
    logger.info("Model loaded successfully!")
    
except Exception as e:
    logger.error(f"Failed to load vLLM model: {e}")
    MODEL_LOADED = False
    llm = None
    sampling_params = None

def format_chat_prompt(user_message: str) -> str:
    """Format prompt for Llama 3.1 chat template"""
    system_prompt = "You are a helpful assistant."
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def generate_response(user_message: str) -> str:
    """Generate response with fallback for compatibility issues"""
    if not MODEL_LOADED or llm is None:
        return f"Model not loaded. Error during initialization. You asked: '{user_message}'"
    
    try:
        # Format the prompt
        formatted_prompt = format_chat_prompt(user_message)
        
        # Try the most basic generation approach
        outputs = llm.generate(formatted_prompt, sampling_params)
        
        # Extract response safely
        if outputs and len(outputs) > 0:
            if hasattr(outputs[0], 'outputs') and len(outputs[0].outputs) > 0:
                response = outputs[0].outputs[0].text.strip()
                return response
            else:
                return "Error: Unexpected output format"
        else:
            return "Error: No output generated"
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}. You asked: '{user_message}'"

# Flask web application
app = Flask(__name__)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Python 3.9 vLLM Chatbot</title>
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
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .status {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            text-align: center;
        }
        .error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
</head>
<body>
    <h1>🤖 Python 3.9 vLLM Chatbot</h1>
    
    <div class="status {{ 'error' if not model_loaded else '' }}">
        {% if model_loaded %}
        ✅ Model loaded successfully! Ready to chat.
        {% else %}
        ❌ Model failed to load. Check console for errors.
        {% endif %}
    </div>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Ask me anything..." 
               onkeypress="if(event.key==='Enter') sendMessage()">
        <button id="sendButton" onclick="sendMessage()">Send</button>
    </div>

    <script>
        let isGenerating = false;
        let currentAssistantDiv = null;

        function sendMessage() {
            if (isGenerating) return;
            
            const input = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Disable input while generating
            isGenerating = true;
            sendButton.disabled = true;
            sendButton.textContent = 'Generating...';
            
            // Create assistant message div for streaming
            currentAssistantDiv = addMessage('', 'assistant');
            
            // Start Server-Sent Events connection
            const eventSource = new EventSource('/stream?message=' + encodeURIComponent(message));
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.token) {
                    // Add token to current message
                    currentAssistantDiv.textContent += data.token;
                    scrollToBottom();
                } else if (data.done) {
                    // Streaming finished
                    eventSource.close();
                    isGenerating = false;
                    sendButton.disabled = false;
                    sendButton.textContent = 'Send';
                    input.focus();
                    currentAssistantDiv = null;
                }
            };
            
            eventSource.onerror = function(event) {
                console.error('EventSource failed:', event);
                eventSource.close();
                
                if (currentAssistantDiv && currentAssistantDiv.textContent === '') {
                    currentAssistantDiv.textContent = 'Error: Could not get response';
                }
                
                isGenerating = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                input.focus();
                currentAssistantDiv = null;
            };
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.textContent = text;
            container.appendChild(div);
            scrollToBottom();
            return div;
        }
        
        function scrollToBottom() {
            const container = document.getElementById('chatContainer');
            container.scrollTop = container.scrollHeight;
        }
        
        // Add welcome message
        window.onload = function() {
            if ({{ model_loaded | tojson }}) {
                addMessage('Hello! I\\'m ready to answer your questions. How can I help you today?', 'assistant');
            } else {
                addMessage('Sorry, there was an error loading the model. Please check the console logs.', 'assistant');
            }
            document.getElementById('messageInput').focus();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template_string(HTML_TEMPLATE, model_loaded=MODEL_LOADED)

@app.route('/stream')
def stream():
    """Stream the AI response"""
    user_message = request.args.get('message', '')
    
    if not user_message:
        return Response("data: " + json.dumps({"error": "No message provided"}) + "\n\n", 
                       mimetype='text/plain')
    
    def generate():
        try:
            logger.info(f"Generating response for: {user_message}")
            
            # Generate response
            response = generate_response(user_message)
            
            # Split response into words and stream them
            words = response.split()
            
            for i, word in enumerate(words):
                if i == 0:
                    token = word
                else:
                    token = " " + word
                
                # Send each word with a small delay for streaming effect
                yield f"data: {json.dumps({'token': token})}\\n\\n"
                time.sleep(0.05)  # Small delay between words
            
            # Signal completion
            yield f"data: {json.dumps({'done': True})}\\n\\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\\n\\n"
            yield f"data: {json.dumps({'done': True})}\\n\\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": MODEL_LOADED, "python_version": "3.9"}

if __name__ == '__main__':
    print("\\n" + "="*50)
    print("🐍 Python 3.9 Compatible vLLM Chatbot")
    print(f"📊 Model loaded: {MODEL_LOADED}")
    print("📍 Open your browser to: http://localhost:8000")
    print("💬 Start chatting!")
    print("="*50 + "\\n")
    
    # Run the server on port 8000
    app.run(host='0.0.0.0', port=8000, debug=False)
