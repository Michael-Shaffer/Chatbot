#!/usr/bin/env python3
"""
Persistent vLLM Document Q&A Web Server
Run this to create a web API that people can access
"""

import os
import json
import logging
from typing import Dict, List
from pathlib import Path

# Disable vLLM tracking
os.environ["VLLM_DO_NOT_TRACK"] = "1"

from flask import Flask, request, jsonify, render_template_string
from vllm import LLM, SamplingParams
import pymupdf4llm
import chromadb
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentQAServer:
    def __init__(self, model_path: str):
        """Initialize the vLLM model and document processing"""
        logger.info("Loading vLLM model...")
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_path,
            quantization="awq",
            dtype="float16",
            max_model_len=4096,
            gpu_memory_utilization=0.6
        )
        
        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]
        )
        
        # Initialize document search (optional - for uploaded docs)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = None
        
        logger.info("Model loaded successfully!")
    
    def format_chat_prompt(self, system_prompt: str, user_message: str, context: str = "") -> str:
        """Format prompt for Llama 3.1 chat template"""
        if context:
            system_content = f"{system_prompt}\n\nRelevant context:\n{context}"
        else:
            system_content = system_prompt
            
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def generate_response(self, user_message: str, system_prompt: str = None) -> str:
        """Generate response using vLLM"""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers questions based on documents and general knowledge."
        
        # Format the prompt
        formatted_prompt = self.format_chat_prompt(system_prompt, user_message)
        
        # Generate response
        outputs = self.llm.generate(formatted_prompt, self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        return response
    
    def add_document(self, file_path: str, doc_name: str = None) -> bool:
        """Add a PDF document to the knowledge base"""
        try:
            # Extract text from PDF
            text = pymupdf4llm.to_markdown(file_path)
            
            # Create collection if doesn't exist
            if self.collection is None:
                self.collection = self.chroma_client.create_collection("documents")
            
            # Add to vector database
            doc_id = doc_name or Path(file_path).stem
            self.collection.add(
                documents=[text],
                ids=[doc_id],
                metadatas=[{"source": file_path, "name": doc_name or doc_id}]
            )
            
            logger.info(f"Added document: {doc_name or doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 3) -> str:
        """Search documents for relevant context"""
        if self.collection is None:
            return ""
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results['documents'] and results['documents'][0]:
                return "\n\n".join(results['documents'][0])
            return ""
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return ""

# Initialize the server
MODEL_PATH = "./Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Update this path
qa_server = DocumentQAServer(MODEL_PATH)

# Flask web application
app = Flask(__name__)

# Simple HTML interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Document Q&A Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin: 20px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .assistant { background-color: #f5f5f5; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; }
        .upload-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Document Q&A Chatbot</h1>
    
    <div class="upload-section">
        <h3>Upload Document (Optional)</h3>
        <input type="file" id="fileInput" accept=".pdf">
        <button onclick="uploadFile()">Upload PDF</button>
        <div id="uploadStatus"></div>
    </div>
    
    <div class="chat-container" id="chatContainer"></div>
    
    <div>
        <input type="text" id="messageInput" placeholder="Ask a question..." 
               onkeypress="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                
                const data = await response.json();
                addMessage(data.response, 'assistant');
            } catch (error) {
                addMessage('Error: Could not get response', 'assistant');
            }
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('chatContainer');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('uploadStatus').textContent = data.message;
            } catch (error) {
                document.getElementById('uploadStatus').textContent = 'Upload failed';
            }
        }
        
        // Add welcome message
        addMessage('Hello! I can answer questions about documents and general topics. Upload a PDF to ask questions about it!', 'assistant');
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
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Search documents for context (if any uploaded)
        context = qa_server.search_documents(user_message)
        
        # Generate response
        if context:
            system_prompt = "You are a helpful assistant. Answer questions based on the provided context and your general knowledge. If the context doesn't contain relevant information, use your general knowledge."
            response = qa_server.generate_response(user_message, system_prompt)
        else:
            response = qa_server.generate_response(user_message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            
            # Add to knowledge base
            success = qa_server.add_document(temp_path, file.filename)
            
            if success:
                return jsonify({'message': f'Successfully uploaded {file.filename}'})
            else:
                return jsonify({'error': 'Failed to process file'}), 500
        else:
            return jsonify({'error': 'Only PDF files are supported'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Add any PDFs in current directory on startup
    pdf_files = list(Path('.').glob('*.pdf'))
    for pdf_file in pdf_files:
        qa_server.add_document(str(pdf_file))
    
    print("\n" + "="*50)
    print("🤖 Document Q&A Chatbot Server Starting...")
    print("📍 Web interface: http://localhost:5000")
    print("📄 Upload PDFs through the web interface")
    print("💬 Chat with your documents!")
    print("="*50 + "\n")
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)
