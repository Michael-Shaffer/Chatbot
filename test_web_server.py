#!/usr/bin/env python3
"""
Simple Flask Chatbot with Streaming Responses
No documents, no ChromaDB - just pure chat with streaming
"""

import os
import time
import json
import logging

# Disable vLLM tracking
os.environ["VLLM_DO_NOT_TRACK"] = "1"

from flask import Flask, request, render_template_string, Response
from vllm import LLM, SamplingParams

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize vLLM model
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Update this path

logger.info("Loading vLLM model...")
llm = LLM(
    model=MODEL_PATH,
    quantization="awq",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.6
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "<|im_end|>", "<|endoftext|>"]
)

logger.info("Model loaded successfully!")

def format_chat_prompt(user_message: str) -> str:
    """Format prompt for Llama 3.1 chat template"""
    system_prompt = "You are a helpful assistant."
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Flask web application
app = Flask(__name__)

# HTML template with Server-Sent Events for streaming
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Streaming AI Chatbot</title>
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
        .typing {
            opacity: 0.7;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>🤖 Streaming AI Chatbot</h1>
    
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
            addMessage('Hello! I\\'m ready to answer your questions. Ask me anything!', 'assistant');
            document.getElementById('messageInput').focus();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the chat interface"""
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
            logger.info(f"Streaming response for: {user_message}")
            
            # Format the prompt
            formatted_prompt = format_chat_prompt(user_message)
            
            # Generate response
            outputs = llm.generate([formatted_prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
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
    return {"status": "healthy", "model": "loaded"}

if __name__ == '__main__':
    print("\\n" + "="*50)
    print("🤖 Streaming AI Chatbot Server Starting...")
    print("📍 Open your browser to: http://localhost:8000")
    print("💬 Watch responses stream in real-time!")
    print("="*50 + "\\n")
    
    # Run the server on port 8000
    app.run(host='0.0.0.0', port=8000, debug=False)
