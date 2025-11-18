import requests

# 🔧 Your system prompt
SYSTEM_PROMPT = """
You are FamilyCare — a kind, helpful, and accurate AI assistant that provides
information about family planning, reproductive health, and contraceptive methods.

Rules:
• Focus only on family planning and reproductive health topics.
• If a question is off-topic, politely say it’s outside your scope.
• Always reply in the same language the user uses.
• If the user asks in Nepali, reply completely in natural Nepali language.
• Do not translate Nepali questions into English.
• Keep your tone friendly, clear, and short.

"""

def chat_with_mistral(user_input):
    """
    Sends the user's input and system prompt to the local Ollama Mistral model.
    """

    # Combine the system prompt + user message
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"

    # Send request to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": full_prompt,
            "stream": True  # stream=True makes it show responses as they generate
        },
        stream=True
    )

    # Print the streamed response
    print("\nFamilyCare:", end=" ", flush=True)
    for line in response.iter_lines():
        if line:
            # Decode line safely
            data = line.decode("utf-8")
            # Only print text tokens, ignore other data
            if '"response"' in data:
                part = data.split('"response":"')[-1].split('"')[0]
                print(part, end="", flush=True)
    print("\n")


# 🧩 Simple interactive loop
if __name__ == "__main__":
    print("👩‍⚕️ FamilyCare Chatbot (powered by Mistral 7B via Ollama)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        chat_with_mistral(user_input)
