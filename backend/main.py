import fastapi
import uvicorn
import whisper
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, AutoModelForAudioClassification
import asyncio
import librosa
import io
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
# import logging
from loguru import logger
from typing import Dict, Any


from utils import load_and_preprocess_audio

# --- Basic Configuration ---
# Select Whisper model size: tiny, base, small, medium, large, large-v2, large-v3
WHISPER_MODEL_SIZE = "base" # Use "base" for a balance of speed and accuracy
# --- UPDATED MODEL ID FOR VALENCE, AROUSAL, DOMINANCE ---
VAD_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-valence-arousal-dominance-msp-dim"
TARGET_SAMPLE_RATE = 16000 # Required sample rate for both models

# --- FastAPI App Setup ---
app = fastapi.FastAPI()

# --- Global Variables for Models and Executor ---
# These will be loaded during startup
whisper_model = None
vad_processor = None # Renamed from emotion_processor
vad_model = None     # Renamed from emotion_model
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use a ThreadPoolExecutor to run synchronous, CPU/GPU-bound tasks
executor = ThreadPoolExecutor()

# --- Model Loading Logic ---
def load_models():
    """Loads the Whisper and VAD models into memory."""
    global whisper_model, vad_processor, vad_model # Updated names
    logger.info(f"Loading models onto device: {device}")
    try:
        # Load Whisper model
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logger.info("Whisper model loaded successfully.")

        # Load VAD model from Hugging Face
        logger.info(f"Loading VAD model: {VAD_MODEL_ID}") # Updated log message
        vad_processor = AutoProcessor.from_pretrained(VAD_MODEL_ID) # Use new ID
        vad_model = AutoModelForAudioClassification.from_pretrained(VAD_MODEL_ID).to(device) # Use new ID

        # Sanity check: VAD models usually output 3 values
        if hasattr(vad_model.config, 'num_labels') and vad_model.config.num_labels == 3:
             logger.info(f"VAD model loaded successfully. Output dimensions: {vad_model.config.num_labels}")
        else:
             num_labels = getattr(vad_model.config, 'num_labels', 'N/A')
             logger.warning(f"Loaded VAD model has {num_labels} output dimensions. Expected 3 for V, A, D. Check model compatibility.")

    except Exception as e:
        logger.error(f"Fatal error loading models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load models: {e}")


def run_whisper_transcription(audio_np: np.ndarray) -> str:
    """Synchronous function to perform Whisper transcription. (Identical to previous version)"""
    if whisper_model is None:
        logger.error("Whisper model not available for transcription.")
        raise RuntimeError("Whisper model is not loaded.")
    logger.info("Starting Whisper transcription...")
    try:
        result = whisper_model.transcribe(audio_np, fp16=False if device == 'cpu' else True)
        transcription = result['text']
        logger.info("Whisper transcription finished.")
        return transcription
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
        return f"Error during transcription: {e}"

def run_vad_analysis(audio_np: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """Synchronous function to perform Valence, Arousal, Dominance analysis."""
    if vad_processor is None or vad_model is None: # Check updated names
        logger.error("VAD model/processor not available for analysis.")
        raise RuntimeError("VAD model or processor is not loaded.")
    logger.info("Starting VAD (Valence, Arousal, Dominance) analysis...")
    try:
        inputs = vad_processor( # Use updated processor
            audio_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Perform inference
        with torch.no_grad():
            # For VAD regression, the model's output logits often directly
            # represent the continuous scores. We don't use softmax.
            logits = vad_model(**inputs).logits # Use updated model

        # Check if the output shape is as expected (batch_size, 3)
        if logits.shape[1] != 3:
             logger.error(f"VAD model output has unexpected shape: {logits.shape}. Expected 3 dimensions for V, A, D.")
             # Provide info about the actual shape in the error
             return {"error": f"VAD model output dimension mismatch. Expected 3, got {logits.shape[1]}"}

        # Extract scores (assuming order V, A, D - check model card if necessary)
        scores = logits[0].cpu().numpy() # Get scores for the first item, move to CPU, convert to numpy

        # Map scores to dictionary
        # Note: The range of these scores depends on the model's training data (e.g., [-1, 1], [0, 1], [1, 5]).
        # This model (trained on MSP-Podcast) likely outputs values roughly in the [1, 5] or similar range,
        # but consult the model card or experiment if precise range is needed.
        results = {
            "valence": float(scores[0]), # Valence score
            "arousal": float(scores[1]), # Arousal score
            "dominance": float(scores[2]), # Dominance score
        }
        logger.info(f"VAD analysis finished. Scores: {results}")
        return results
    except Exception as e:
        logger.error(f"Error during VAD analysis: {e}", exc_info=True)
        return {"error": f"VAD analysis failed: {e}"}


# --- API Endpoint ---
@app.post("/process_audio/")
async def process_audio_endpoint(file: fastapi.UploadFile = fastapi.File(...)) -> Dict[str, Any]:
    """
    Accepts an audio file upload, performs Whisper transcription and
    Valence-Arousal-Dominance (VAD) analysis in parallel.

    Returns:
        A JSON object containing the filename, transcription text,
        and a dictionary of VAD scores (valence, arousal, dominance).
    """
    logger.info(f"Received request for file: {file.filename}, content type: {file.content_type}")

    if not file.content_type or not file.content_type.startswith("audio/"):
         logger.warning(f"Invalid content type received: {file.content_type}")
         raise fastapi.HTTPException(
             status_code=fastapi.status.HTTP_400_BAD_REQUEST,
             detail=f"Invalid file type '{file.content_type}'. Please upload an audio file."
         )

    # Check if models are loaded
    if whisper_model is None or vad_model is None: # Check VAD model
         logger.error("Attempted to process audio, but models are not loaded.")
         raise fastapi.HTTPException(
              status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
              detail="Models are not loaded or failed to load. Service unavailable."
         )

    try:
        # 1. Read file content
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from '{file.filename}'.")

        # 2. Preprocess audio
        audio_np = await load_and_preprocess_audio(file_content, target_sr=TARGET_SAMPLE_RATE)

        # 3. Schedule synchronous tasks in the thread pool executor
        loop = asyncio.get_running_loop()
        logger.info("Scheduling transcription and VAD analysis tasks in parallel...")
        transcription_task = loop.run_in_executor(executor, run_whisper_transcription, audio_np)
        # --- Schedule VAD task ---
        vad_task = loop.run_in_executor(executor, run_vad_analysis, audio_np, TARGET_SAMPLE_RATE) # Call updated function

        # 4. Await completion of both tasks concurrently
        results = await asyncio.gather(
            transcription_task,
            vad_task, # Await updated task
            return_exceptions=True # Return exceptions instead of raising
        )
        logger.info("Both tasks completed.")

        # 5. Process results and handle potential errors
        transcription_result = results[0]
        vad_result = results[1] # Updated variable name

        # Handle transcription errors
        if isinstance(transcription_result, Exception):
             logger.error(f"Transcription task failed for {file.filename}: {transcription_result}")
             transcription_result = f"Transcription Error: {transcription_result}"
             # Consider if this should be a fatal error for the request

        # Handle VAD analysis errors
        if isinstance(vad_result, Exception):
             logger.error(f"VAD analysis task failed for {file.filename}: {vad_result}")
             vad_result = {"error": f"VAD Analysis Error: {vad_result}"}
             # Consider if this should be a fatal error

        # 6. Return combined results with updated key
        return {
            "filename": file.filename,
            "transcription": transcription_result,
            "vad_scores": vad_result # --- UPDATED RESPONSE KEY ---
        }

    except ValueError as ve: # Catch audio processing errors
         logger.error(f"Audio processing error for {file.filename}: {ve}", exc_info=True)
         raise fastapi.HTTPException(
             status_code=fastapi.status.HTTP_400_BAD_REQUEST,
             detail=f"Error processing audio: {ve}"
         )
    except RuntimeError as re: # Catch model loading issues surfaced during request
        logger.error(f"Model runtime error processing {file.filename}: {re}", exc_info=True)
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model runtime error: {re}"
        )
    except Exception as e:
        # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred processing {file.filename}: {e}", exc_info=True)
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )

# --- Run Server (for local testing) ---
if __name__ == "__main__":
    # Recommended command: uvicorn main:app --host 0.0.0.0 --port 8000 [--reload]
    logger.info("Starting Uvicorn server directly...")

    load_models()  
    logger.info("Models loaded successfully. Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)