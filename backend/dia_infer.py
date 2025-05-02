#!/usr/bin/env python3
import os
import soundfile as sf
import torch
from dia.model import Dia

# 1) Load the Dia model once at import
MODEL = Dia.from_pretrained("nari-labs/Dia-1.6B")


def infer_tts(
    text: str,
    output_path: str = "output.wav",
    audio_prompt: str = None,
    sample_rate: int = 22050,
    **generate_kwargs
) -> str:
    """
    Generate speech from text using the Dia TTS model.

    Args:
        text: Transcript to synthesize.
        output_path: Where to write the .wav file.
        audio_prompt: Optional path to a .wav file to condition voice/style.
        sample_rate: Sampling rate for the output file.
        generate_kwargs: Extra generation parameters (e.g., temperature).

    Returns:
        The filepath of the written WAV.
    """
    # If an audio prompt is given, load it to a NumPy array
    if audio_prompt:
        prompt_wav, _ = sf.read(audio_prompt)
        wav = MODEL.generate(text, audio_prompt=prompt_wav, **generate_kwargs)
    else:
        wav = MODEL.generate(text, **generate_kwargs)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Write the WAV file
    sf.write(output_path, wav, sample_rate)
    return output_path


def main():
    """
    Simple test harness for infer_tts().
    Modify the sample_text or parameters below to try different inputs.
    """
    sample_text = "[S1] Hello, world! (laughs) [S2] Hi there! Nice to meet you."
    print("Generating speech for:", sample_text)
    out_file = "demo_output.wav"

    # Optional: if you have a reference wav for voice cloning, set audio_prompt to its path
    audio_prompt = None  # e.g., "actor_reference.wav"

    # Invoke the inference function
    generated = infer_tts(
        text=sample_text,
        output_path=out_file,
        audio_prompt=audio_prompt,
        temperature=0.7,    # you can tweak temperature, max_length, etc.
    )

    print("✔ Generated audio at:", generated)


if __name__ == "__main__":
    main()

