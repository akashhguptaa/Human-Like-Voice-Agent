import fastapi
import uvicorn
import whisper
import torch
import torch.nn as nn  # Added
import numpy as np
from transformers import (  # Added specific imports
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# Auto* classes no longer needed for VAD model
import asyncio
import librosa
import io
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, Any

# --- Basic Configuration ---
WHISPER_MODEL_SIZE = "base"
# Ensure this matches the model you intend to use (VAD one)
VAD_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-valence-arousal-dominance-msp-dim"
TARGET_SAMPLE_RATE = 16000

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = fastapi.FastAPI()

# --- Global Variables ---
whisper_model = None
vad_processor: Wav2Vec2Processor = None  # Type hint for clarity
vad_model: Wav2Vec2PreTrainedModel = (
    None  # Type hint for clarity (EmotionModel inherits this)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
executor = ThreadPoolExecutor()

# --- Custom Model Class Definitions (Copied from your example) ---


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Ensure handling matches Wav2Vec2Model expectations
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        # Pooling: Average across the time dimension as per example
        hidden_states_pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states_pooled)

        # Return tuple consistent with the user-provided example `process_func`
        # (pooled hidden states, logits)
        return (hidden_states_pooled, logits)


# --- Model Loading Logic (Updated) ---
def load_models():
    """Loads the Whisper and VAD models into memory using specified classes."""
    global whisper_model, vad_processor, vad_model
    logger.info(f"Loading models onto device: {device}")
    # --- IMPORTANT: Authentication ---
    # Ensure you have run `huggingface-cli login` OR provide token explicitly.
    # Set use_auth_token=True to use token from CLI login (recommended).
    # Replace with `token="hf_YOUR_TOKEN"` if passing token directly.
    auth_token_arg = {"use_auth_token": True}

    try:
        # Load Whisper model (remains the same)
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logger.info("Whisper model loaded successfully.")

        # --- Load VAD model using specific classes and authentication ---
        logger.info(f"Loading VAD model: {VAD_MODEL_ID} using specified classes")
        vad_processor = Wav2Vec2Processor.from_pretrained(
            VAD_MODEL_ID, **auth_token_arg
        )
        # Use the custom EmotionModel class for loading
        vad_model = EmotionModel.from_pretrained(VAD_MODEL_ID, **auth_token_arg).to(
            device
        )
        logger.info(
            "VAD processor and model loaded successfully using specified classes."
        )

    except Exception as e:
        logger.error(f"Fatal error loading models: {e}", exc_info=True)
        if "401" in str(e) or "Repository Not Found" in str(e):
            logger.error(
                "Authentication Error: Failed to load VAD model."
                " Please ensure you have:"
                " 1. Visited the model page on Hugging Face and accepted the terms."
                " 2. Logged in via `huggingface-cli login` in your terminal."
                " 3. Or correctly passed a valid token if using the 'token=' argument."
            )
        # Re-raise the exception to stop the application if models fail to load
        raise RuntimeError(f"Failed to load models: {e}")


# --- FastAPI Startup Event --- (Remains the same)
@app.on_event("startup")
async def startup_event():
    """Loads models when the FastAPI application starts."""
    loop = asyncio.get_running_loop()
    # Run blocking model load in executor thread
    await loop.run_in_executor(executor, load_models)
    if whisper_model is None or vad_model is None:
        # This case should ideally be covered by the exception in load_models
        logger.warning(
            "One or both models failed to load. Endpoints may not function correctly."
        )


# --- Helper Function for Audio Processing --- (Remains the same)
async def load_and_preprocess_audio(file_content: bytes, target_sr: int) -> np.ndarray:
    # ... (Identical to previous versions) ...
    logger.info(f"Processing audio data ({len(file_content)} bytes)...")
    try:
        audio_data, sr = sf.read(io.BytesIO(file_content))
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            logger.info("Audio is stereo, converting to mono.")
            if audio_data.shape[1] > audio_data.shape[0]:
                audio_data = audio_data.mean(axis=1)
            else:
                audio_data = audio_data.mean(axis=0)
        elif audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        if sr != target_sr:
            logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz.")
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

        if audio_data.dtype != np.float32:
            logger.info(
                f"Converting audio data type from {audio_data.dtype} to float32."
            )
            audio_data = audio_data.astype(np.float32)

        logger.info(
            f"Audio processed: length={len(audio_data)} samples, sample_rate={target_sr}, dtype={audio_data.dtype}"
        )
        return audio_data
    except Exception as e:
        logger.error(f"Error processing audio file: {e}", exc_info=True)
        raise ValueError(f"Could not process audio file: {e}")


# --- Synchronous Task Functions ---


# Whisper function (Remains the same)
def run_whisper_transcription(audio_np: np.ndarray) -> str:
    # ... (Identical to previous versions) ...
    if whisper_model is None:
        logger.error("Whisper model not available for transcription.")
        raise RuntimeError("Whisper model is not loaded.")
    logger.info("Starting Whisper transcription...")
    try:
        result = whisper_model.transcribe(
            audio_np, fp16=False if device == "cpu" else True
        )
        transcription = result["text"]
        logger.info("Whisper transcription finished.")
        return transcription
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
        return f"Error during transcription: {e}"


# --- UPDATED VAD ANALYSIS FUNCTION (using provided structure) ---
def run_vad_analysis(audio_np: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """Synchronous function to perform VAD analysis using the specified model structure."""
    if vad_processor is None or vad_model is None:
        logger.error("VAD model/processor not available for analysis.")
        raise RuntimeError("VAD model or processor is not loaded.")
    logger.info("Starting VAD (Arousal, Dominance, Valence) analysis...")
    try:
        # 1. Process audio using the Wav2Vec2Processor
        #    Match the input preparation from the user's example `process_func`
        #    Processor handles normalization, padding, and returns tensors
        processed_inputs = vad_processor(
            audio_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,  # Ensure sequences are padded to the same length if needed
        )
        input_values = processed_inputs.input_values.to(
            device
        )  # Shape: (batch=1, sequence_length)

        # Include attention_mask if the processor provides it (recommended)
        attention_mask = processed_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 2. Perform inference using the custom EmotionModel
        with torch.no_grad():
            # The custom model's forward returns a tuple: (hidden_states_pooled, logits)
            # We need the logits (the second element, index 1)
            hidden_states_pooled, logits = vad_model(
                input_values=input_values, attention_mask=attention_mask
            )

        # 3. Extract scores and map to labels based on documentation order
        #    Logits shape should be (batch_size=1, 3)
        if logits.shape[0] != 1 or logits.shape[1] != 3:
            logger.error(
                f"VAD model output logits have unexpected shape: {logits.shape}. Expected (1, 3)."
            )
            return {
                "error": f"VAD model output dimension mismatch. Expected (1, 3), got {logits.shape}"
            }

        scores = (
            logits[0].cpu().numpy()
        )  # Get scores for the first (only) item -> shape (3,)

        # --- Map scores based on the documented order: Arousal, Dominance, Valence ---
        results = {
            "arousal": float(scores[0]),  # Index 0 = Arousal
            "dominance": float(scores[1]),  # Index 1 = Dominance
            "valence": float(scores[2]),  # Index 2 = Valence
        }
        logger.info(f"VAD analysis finished. Scores (A, D, V): {results}")
        return results
    except Exception as e:
        logger.error(f"Error during VAD analysis: {e}", exc_info=True)
        return {"error": f"VAD analysis failed: {e}"}


# --- API Endpoint --- (Remains the same, calls updated run_vad_analysis)
@app.post("/process_audio/")
async def process_audio_endpoint(
    file: fastapi.UploadFile = fastapi.File(...),
) -> Dict[str, Any]:
    # ... (Identical structure to previous versions, calls run_whisper_transcription and run_vad_analysis) ...
    logger.info(
        f"Received request for file: {file.filename}, content type: {file.content_type}"
    )

    if not file.content_type or not file.content_type.startswith("audio/"):
        logger.warning(f"Invalid content type received: {file.content_type}")
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Please upload an audio file.",
        )

    if whisper_model is None or vad_model is None:
        logger.error("Attempted to process audio, but models are not loaded.")
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded or failed to load. Service unavailable.",
        )

    try:
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from '{file.filename}'.")
        audio_np = await load_and_preprocess_audio(
            file_content, target_sr=TARGET_SAMPLE_RATE
        )

        loop = asyncio.get_running_loop()
        logger.info("Scheduling transcription and VAD analysis tasks in parallel...")
        transcription_task = loop.run_in_executor(
            executor, run_whisper_transcription, audio_np
        )
        vad_task = loop.run_in_executor(
            executor, run_vad_analysis, audio_np, TARGET_SAMPLE_RATE
        )

        results = await asyncio.gather(
            transcription_task, vad_task, return_exceptions=True
        )
        logger.info("Both tasks completed.")

        transcription_result = results[0]
        vad_result = results[1]

        if isinstance(transcription_result, Exception):
            logger.error(
                f"Transcription task failed for {file.filename}: {transcription_result}"
            )
            transcription_result = f"Transcription Error: {transcription_result}"

        if isinstance(vad_result, Exception):
            logger.error(f"VAD analysis task failed for {file.filename}: {vad_result}")
            # Check if the error is already a dict (e.g., from inside run_vad_analysis)
            if not isinstance(vad_result, dict):
                vad_result = {"error": f"VAD Analysis Error: {vad_result}"}

        return {
            "filename": file.filename,
            "transcription": transcription_result,
            "vad_scores": vad_result,
        }

    except ValueError as ve:
        logger.error(f"Audio processing error for {file.filename}: {ve}", exc_info=True)
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing audio: {ve}",
        )
    except RuntimeError as re:
        logger.error(
            f"Model runtime error processing {file.filename}: {re}", exc_info=True
        )
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model runtime error: {re}",
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred processing {file.filename}: {e}",
            exc_info=True,
        )
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}",
        )


# --- Run Server --- (Remains the same)
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
