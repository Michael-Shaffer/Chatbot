import requests
import json

VLLM_URL = "http://localhost:8000/v1/chat/completions"

def run_rag_pipeline(user_query):
    print("RAG Pipeline: Received query ->", user_query)
    # RAG context retrieval would happen here

    # 1. Add "stream": True to the payload
    payload = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [
            {"role": "user", "content": user_query}
        ],
        "stream": True  # This is the key to enable streaming
    }

    try:
        # 2. Add stream=True to the request call
        response = requests.post(VLLM_URL, json=payload, stream=True)
        response.raise_for_status()

        # 3. Iterate over the streaming response
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # vLLM streams Server-Sent Events (SSE). We need to parse them.
                if decoded_line.startswith('data: '):
                    # Remove the 'data: ' prefix
                    json_str = decoded_line[6:]
                    # Handle the final "[DONE]" message
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(json_str)
                        # Extract the token and yield it
                        token = data['choices'][0]['delta'].get('content', '')
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        print(f"Could not decode line: {json_str}")

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to vLLM server. Details: {e}")
        yield "Sorry, I was unable to connect to the language model."
