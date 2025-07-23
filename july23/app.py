# No longer need session, stream_with_context for this route
from flask import Flask, request, Response 

@app.route("/api/chat", methods=["POST"])
def chat_api():
    # 1. Get the entire history from the client
    messages = request.json.get("messages", [])

    # (Optional) Add a print statement to see what the backend receives
    # print(json.dumps(messages, indent=2))

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/devfiles/git_repos_big2/shaffem/polaris/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "messages": messages, // <-- Use the history from the request
        "stream": True
    }

    def stream_response():
        # This generator now only needs to stream. No session logic.
        try:
            response = requests.post(URL, headers=headers, data=json.dumps(payload), stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                # ... (your existing logic to parse and yield chunks) ...
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to VLLM: {e}")
            yield "Error connecting to the model."

    return Response(stream_response(), mimetype='text/plain')
