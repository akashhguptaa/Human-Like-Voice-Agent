import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

ELEVENLABS_API_KEY = (
    os.getenv("ELEVEN_LABS_KEY")  # Replace this
)


def get_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        voices = response.json().get("voices", [])
        if voices:
            for v in voices:
                print(f"Voice: {v['name']} - ID: {v['voice_id']}")
            return voices
        else:
            print("❌ No voices found in your account.")
            return []
    else:
        print("❌ Failed to fetch voices:", response.status_code, response.text)
        return []


def map_vad_to_settings(valence, arousal):
    valence = max(0, min(1, valence))
    arousal = max(0, min(1, arousal))
    similarity_boost = round(valence, 2)
    stability = round(1 - arousal, 2)
    return stability, similarity_boost


def process_llama_response(llama_response):
    try:
        response_data = json.loads(llama_response)
        sentences = []
        for sentence_key in response_data.get("response", {}):
            sentence_data = response_data["response"][sentence_key]
            sentences.append(sentence_data["text"])
        return " ".join(sentences)
    except json.JSONDecodeError as e:
        print(f"Error parsing Llama response: {e}")
        return llama_response


def text_to_speech_vad(llama_response, voice_id, filename="output.wav"):
    # Process the Llama response to get clean text
    text = process_llama_response(llama_response)

    # Default VAD settings
    stability, similarity_boost = 0.5, 0.5

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio saved to {filename}")
        return {"status": "success", "filename": filename}
    else:
        print("❌ Error:", response.status_code)
        print(response.text)
        return {"status": "error", "message": response.text}


# Example usage
if __name__ == "__main__":
    voices = get_voices()
    if voices:
        voice_id = voices[0]["voice_id"]
        sample_response = '{"response": {"s1": {"text": "Hello, how are you?", "vad": [0.7, 0.4, 0.6]}}}'
        text_to_speech_vad(sample_response, voice_id)
    else:
        print("Please add a voice in your ElevenLabs dashboard first.")
