@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_message = request.json.get("query", "")

    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({"role": "user", "content": user_message})
    
    # Keep the history from getting too long
    session["chat_history"] = session["chat_history"][-10:] 

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/devfiles/git_repos_big2/shaffem/polaris/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "messages": session["chat_history"],
        "stream": True
    }

    # Debugging: Print the payload being sent to the LLM
    print("--- Sending to LLM: ---")
    print(json.dumps(payload, indent=2))
    print("-----------------------")

    def stream_response():
        """This generator streams the response and saves it to history."""
        full_response = ""
        try:
            # The stream_with_context wrapper keeps this session alive
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
            
            # --- CRITICAL FIX ---
            # This part runs after the stream is complete.
            # It saves the full response to the session history.
            session["chat_history"].append({"role": "assistant", "content": full_response})
            print("--- Assistant response saved. New history length:", len(session["chat_history"]), "---")


        except requests.exceptions.RequestException as e:
            print(f"Error connecting to VLLM: {e}")
            yield "Error connecting to the model."

    return Response(stream_with_context(stream_response()), mimetype='text/plain')
