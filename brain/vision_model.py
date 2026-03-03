import ollama

# 1. Initialize the memory with your fixed system rules
messages = [
    {'role': 'system', 'content': 'You are my personal AI assistant called TABS. My name is Mohammed Saqib Junaid Khan. You call me Sir.'}
]

while True:
    x = input("User: ")
    if x.lower() in ['exit', 'quit']: break

    # 2. Add the new user message to the memory
    messages.append({'role': 'user', 'content': x})

    # 3. Send the entire history (memory) to the model
    stream = ollama.chat(
        model='qwen2.5:3b', 
        messages=messages, 
        stream=True,
        options={'num_predict': 2042} # Your token cap
    )

    print("AI: ", end="", flush=True)
    full_response = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        full_response += content
    print() # New line after the response
    
    # 4. Save the AI's response to the memory so it can refer back to it
    messages.append({'role': 'assistant', 'content': full_response})
