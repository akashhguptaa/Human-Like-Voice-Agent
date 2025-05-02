#!/usr/bin/env python3
"""
Module providing a simple function to run inference with Nari Labs' Dia TTS model.
Installation:
    pip install git+https://github.com/nari-labs/dia.git
"""
from dia.model import Dia

def infer_dia_tts(
    text: str,
    output_path: str = "output.wav",
    model_id: str = "nari-labs/Dia-1.6B",
    dtype: str = "float16",
    use_compile: bool = True
) -> None:
    """
    Generate speech from input text using the specified Dia TTS model and save to a file.

    Args:
        text: Input string with speaker tags ([S1], [S2]).
        output_path: File path to save the generated audio.
        model_id: Model identifier (e.g., "nari-labs/Dia-1.6B").
        dtype: Data type for model weights: "float16", "float32", or "bfloat16".
        use_compile: Whether to enable torch.compile optimization.

    Returns:
        None. Audio is written to output_path.
    """
    # Load pretrained model
    model = Dia.from_pretrained(
        model_id,
        compute_dtype=dtype
    )

    # Generate audio tensor
    audio = model.generate(
        text,
        use_torch_compile=use_compile,
        verbose=True
    )

    # Save to file
    model.save_audio(output_path, audio)


# Sample usage (for testing/importing elsewhere)
if __name__ == "__main__":
    sample_text = "Hello, world! (laughs) Hi there! Nice to meet you."
    infer_dia_tts(
        text=sample_text,
        output_path="sample_output.wav",
        model_id="nari-labs/Dia-1.6B",
        dtype="float16",
        use_compile=True
    )
    print("Inference complete. Audio saved to sample_output.wav")

