# In app.py
@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_message = request.json.get("query", "")

    # 1. Initialize or retrieve chat history from the session
    if "chat_history" not in session:
        session["chat_history"] = []

    # 2. Add the new user message to the history
    session["chat_history"].append({"role": "user", "content": user_message})

    # Optional: Keep the history from getting too long (see note below)
    # session["chat_history"] = session["chat_history"][-10:] 

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/devfiles/git_repos_big2/shaffem/polaris/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "messages": session["chat_history"], # <-- Use the full history
        "stream": True
    }

    def stream_response():
        """This generator streams the response and saves it to history."""
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
                                full_response += content # Accumulate the response
                                yield content # Stream the chunk to the frontend
                        except json.JSONDecodeError:
                            continue
            
            # 3. Once the stream is done, add the full response to history
            session["chat_history"].append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to VLLM: {e}")
            yield "Error connecting to the model."

    return Response(stream_response(), mimetype='text/plain')
