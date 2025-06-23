#!/usr/bin/env python3
"""
Simple vLLM Chatbot Web Server
Just ask questions and get answers - no document processing
"""

import os
import logging

# Disable vLLM tracking
os.environ["VLLM_DO_NOT_TRACK"] = "1"

from flask import Flask, request, jsonify, render_template_string, Response
from vllm import LLM, SamplingParams
import json

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

# Simple HTML interface with streaming
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Simple AI Chatbot</title>
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
        .typing {
            opacity: 0.7;
            font-style: italic;
        }
        h1 {
            text-align: center;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>🤖 Simple AI Chatbot</h1>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Ask me anything..." 
               onkeypress="if(event.key==='Enter' && !event.shiftKey) sendMessage()">
        <button id="sendButton" onclick="sendMessage()">Send</button>
    </div>

    <script>
        let isGenerating = false;

        async function sendMessage() {
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
            
            // Add typing indicator
            const typingDiv = addMessage('Thinking...', 'assistant', true);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                // Remove typing indicator
                typingDiv.remove();
                
                if (response.ok) {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    const messageDiv = addMessage('', 'assistant');
                    let fullResponse = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (data.token) {
                                        fullResponse += data.token;
                                        messageDiv.textContent = fullResponse;
                                        scrollToBottom();
                                    }
                                } catch (e) {
                                    // Ignore malformed JSON
                                }
                            }
                        }
                    }
                } else {
                    addMessage('Error: Could not get response', 'assistant');
                }
                
            } catch (error) {
                typingDiv.remove();
                addMessage('Error: Connection failed', 'assistant');
            } finally {
                isGenerating = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                input.focus();
            }
        }
        
        function addMessage(text, sender, isTyping = false) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            if (isTyping) div.className += ' typing';
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
            addMessage('Hello! I\\'m ready to answer your questions. What would you like to know?', 'assistant');
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

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with streaming"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Format the prompt
        formatted_prompt = format_chat_prompt(user_message)
        
        def generate():
            # Generate response
            outputs = llm.generate(formatted_prompt, sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # Stream the response word by word
            words = response.split()
            for i, word in enumerate(words):
                if i == 0:
                    token = word
                else:
                    token = " " + word
                
                yield f"data: {json.dumps({'token': token})}\\n\\n"
            
            # Signal completion
            yield f"data: {json.dumps({'done': True})}\\n\\n"
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

if __name__ == '__main__':
    print("\\n" + "="*50)
    print("🤖 Simple AI Chatbot Server Starting...")
    print("📍 Open your browser to: http://localhost:5000")
    print("💬 Start chatting with your AI!")
    print("="*50 + "\\n")
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)
