from flask import Flask, request, jsonify, send_from_directory, Response # Add Response
import json
from rag_pipeline import run_rag_pipeline
import os

app = Flask(__name__, static_folder='frontend')

# --- API Endpoint ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query not provided"}), 400

    user_query = data['query']

    # This is the new streaming logic
    def stream_response():
        # run_rag_pipeline is now a generator, so we can loop over it
        for token in run_rag_pipeline(user_query):
            # We must format this as a Server-Sent Event for the browser
            yield f"data: {json.dumps({'token': token})}\n\n"

    # Return a streaming response with the correct mimetype
    return Response(stream_response(), mimetype='text/event-stream')

# --- Static File Serving Routes (these stay the same) ---
@app.route('/chat')
def chat_page():
    return send_from_directory(os.path.join(app.static_folder, 'chat'), 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def index_page():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=False)
