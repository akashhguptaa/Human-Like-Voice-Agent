import requests

API_URL = "https://router.huggingface.co/fal-ai/fal-ai/dia-tts"
headers = {
    "Authorization": "Bearer xxxxxxxxxxx",  # Replace with your token
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

payload = {
    "text": "Hello, this is a test using FAL TTS from Hugging Face."
}
print("akash")
result = query(payload)
print(result)  # TEMP: Print to inspect response structure

# If the API returns a URL to the audio
if 'audio' in result and isinstance(result['audio'], dict) and 'url' in result['audio']:
    audio_url = result['audio']['url']
    audio_response = requests.get(audio_url)

    if audio_response.status_code == 200:
        with open("output.wav", "wb") as f:
            f.write(audio_response.content)
        print("Saved audio as output.wav")
    else:
        print("Failed to download audio from URL:", audio_url)
else:
    print("Unexpected response format:", result)

