@app.route("/api/chat", methods=["POST"])
def chat_api():
    # 1. PRINT HISTORY AT THE VERY START OF THE REQUEST
    print("\n--- [START] New Request Received ---")
    if "chat_history" not in session:
        session["chat_history"] = []
    print("Step 1: History at start of request:", json.dumps(session.get("chat_history"), indent=2))

    # 2. ADD USER MESSAGE AND PRINT AGAIN
    user_message = request.json.get("query", "")
    session["chat_history"].append({"role": "user", "content": user_message})
    print("Step 2: History after adding user message:", json.dumps(session.get("chat_history"), indent=2))
    
    # Keep the history from getting too long
    session["chat_history"] = session["chat_history"][-10:] 

    # 3. PREPARE AND PRINT THE PAYLOAD SENT TO THE LLM
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/devfiles/git_repos_big2/shaffem/polaris/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "messages": session["chat_history"],
        "stream": True
    }
    print("Step 3: Payload being sent to LLM:", json.dumps(payload, indent=2))

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
            
            # 4. PRINT THE FINAL RESPONSE AND THE UPDATED HISTORY
            print("Step 4: Captured full assistant response:", full_response)
            updated_history = session["chat_history"] + [{"role": "assistant", "content": full_response}]
            session["chat_history"] = updated_history
            print("Step 5: Final history saved to session:", json.dumps(session.get("chat_history"), indent=2))

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to VLLM: {e}")
            yield "Error connecting to the model."

    return Response(stream_with_context(stream_response()), mimetype='text/plain')
