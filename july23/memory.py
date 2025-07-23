# In app.py

# Make sure to import stream_with_context
from flask import Flask, session, request, Response, stream_with_context

# ... (your other code) ...

@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_message = request.json.get("query", "")

    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({"role": "user", "content": user_message})
    session["chat_history"] = session["chat_history"][-10:] 

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/devfiles/git_repos_big2/shaffem/polaris/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "messages": session["chat_history"],
        "stream": True
    }

    # The logic inside this generator can stay the same
    def stream_response():
        full_response = ""
        try:
            response = requests.post(URL, headers=headers, data=json.dumps(payload), stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(json_str)
                            if chunk['choices'][0]['delta'].get('content'):
                                content = chunk['choices'][0]['delta']['content']
                                full_response += content
                                yield content
                        except json.JSONDecodeError:
                            continue
            
            session["chat_history"].append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to VLLM: {e}")
            yield "Error connecting to the model."

    # This is the key change: wrap the generator with stream_with_context
    return Response(stream_with_context(stream_response()), mimetype='text/plain')
