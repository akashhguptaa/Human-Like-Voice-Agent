import os
import asyncio
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(API_KEY)
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# System prompt
system_prompt = """
You are an emotionally intelligent AI assistant. You deeply understand and respond to human emotions using the VAD (Valence, Arousal, Dominance) model.
You will receive inputs in the following format:
{prompt: user_prompt, VAD: [valence, arousal, dominance]}
Your task is to:
1. Emotionally interpret the user's state based on the VAD values.
2. Respond like a supportive big brother or trustworthy friend—compassionate, warm, reliable.
3. Generate a detailed response with multiple emotionally expressive sentences.
Your response must be formatted as a structured JSON-style object like this:
{
  "response": {
    "res_sentence_1": {
      "text": "Your first sentence here.",
      "vad": [V, A, D]
    },
    ...
  }
}
Assign a unique VAD score for each sentence reflecting the emotional tone of that sentence.
Always keep your tone supportive, warm, and understanding.
"""

# Chat history
chat_history = [{"role": "system", "content": system_prompt.strip()}]

def add_to_history(role, content):
    chat_history.append({"role": role, "content": content})

async def query_model(user_prompt, vad):
    user_input = f"prompt: {user_prompt}, VAD: {vad}"
    add_to_history("user", user_input)
    
    # Initialize the InferenceClient with SambaNova provider
    client = InferenceClient(
        provider="sambanova",
        api_key=API_KEY,
    )
    
    try:
        # Create the completion
        completion = client.chat.completions.create(
            model=MODEL,
            messages=chat_history,
            max_tokens=512,
        )
        
        # Extract the response
        reply = completion.choices[0].message.content
        add_to_history("assistant", reply)
        return reply
    except Exception as e:
        return f"[ERROR] API call failed: {e}"

async def chat():
    print("Starting 7-turn emotional chat with predefined inputs...\n")
    example_inputs = [
        ("I feel really anxious about my exams tomorrow.", [0.2, 0.7, 0.3]),
        ("I think my best friend is ignoring me.", [0.3, 0.6, 0.2]),
        ("I got a promotion at work today!", [0.9, 0.6, 0.8]),
        ("Sometimes I feel like I'm not good enough.", [0.2, 0.5, 0.3]),
        ("I had such a peaceful walk in the park today.", [0.7, 0.3, 0.6]),
        ("I'm nervous about speaking in public tomorrow.", [0.4, 0.8, 0.3]),
        ("I feel hopeful about the future lately.", [0.8, 0.4, 0.7]),
    ]
    
    for i, (user_prompt, vad) in enumerate(example_inputs):
        print(f"\nYou ({i+1}/7): {user_prompt} | VAD: {vad}")
        reply = await query_model(user_prompt, vad)
        print("\nAI Response:\n", reply, "\n")

# Main entry for asyncio to run
if __name__ == "__main__":
    print("Welcome to your Emotionally Intelligent Voice Assistant!")
    asyncio.run(chat())  # Correctly run the chat loop using asyncio