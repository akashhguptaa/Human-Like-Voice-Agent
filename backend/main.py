import fastapi
import uvicorn
import whisper
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from fastapi.middleware.cors import CORSMiddleware
import asyncio

import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Dict, Any
import tempfile
import os
from utils import load_and_preprocess_audio
from llama_VAD import query_model
from dia_tts import query as tts_query
import requests

WHISPER_MODEL_SIZE = "base"
VAD_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TARGET_SAMPLE_RATE = 16000

app = fastapi.FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
whisper_model = None
vad_processor: Wav2Vec2Processor = None
vad_model: Wav2Vec2PreTrainedModel = None
device = "cuda" if torch.cuda.is_available() else "cpu"
executor = ThreadPoolExecutor()


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


def load_models():
    """Loads the Whisper and VAD models into memory using specified classes."""
    global whisper_model, vad_processor, vad_model
    logger.info(f"Loading models onto device: {device}")

    try:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logger.info("Whisper model loaded successfully.")

        logger.info(f"Loading VAD model: {VAD_MODEL_ID} using specified classes")
        vad_processor = Wav2Vec2Processor.from_pretrained(VAD_MODEL_ID)

        vad_model = EmotionModel.from_pretrained(VAD_MODEL_ID).to(device)
        logger.info(
            "VAD processor and model loaded successfully using specified classes."
        )

    except Exception as e:
        logger.error(f"Fatal error loading models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load models: {e}")


def run_whisper_transcription(audio_np: np.ndarray) -> str:
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


def run_vad_analysis(audio_np: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """Synchronous function to perform VAD analysis using the specified model structure."""
    if vad_processor is None or vad_model is None:
        logger.error("VAD model/processor not available for analysis.")
        raise RuntimeError("VAD model or processor is not loaded.")
    logger.info("Starting VAD (Arousal, Dominance, Valence) analysis...")
    try:
        # 1. Process audio using the Wav2Vec2Processor
        processed_inputs = vad_processor(
            audio_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed_inputs.input_values.to(device)

        attention_mask = processed_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 2. Perform inference using the custom EmotionModel
        with torch.no_grad():
            hidden_states_pooled, logits = vad_model(
                input_values=input_values, attention_mask=attention_mask
            )

        # 3. Extract scores and map to labels based on documentation order
        if logits.shape[0] != 1 or logits.shape[1] != 3:
            logger.error(
                f"VAD model output logits have unexpected shape: {logits.shape}. Expected (1, 3)."
            )
            return {
                "error": f"VAD model output dimension mismatch. Expected (1, 3), got {logits.shape}"
            }

        scores = logits[0].cpu().numpy()

        results = {
            "arousal": float(scores[0]),  # 0 = arousal
            "dominance": float(scores[1]),  # 1 = dominance
            "valence": float(scores[2]),  # 2 = vlence
        }
        logger.info(f"VAD analysis finished. Scores (A, D, V): {results}")
        return results
    except Exception as e:
        logger.error(f"Error during VAD analysis: {e}", exc_info=True)
        return {"error": f"VAD analysis failed: {e}"}


@app.post("/process_audio/")
async def process_audio_endpoint(
    file: fastapi.UploadFile = fastapi.File(...),
) -> Dict[str, Any]:
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

        try:
            audio_np = await load_and_preprocess_audio(
                file_content, target_sr=TARGET_SAMPLE_RATE
            )
            logger.info(f"Audio processed successfully")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing audio: {e}",
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
            if not isinstance(vad_result, dict):
                vad_result = {"error": f"VAD Analysis Error: {vad_result}"}

        # Get emotional response from Llama
        emotional_response = await query_model(
            transcription_result,
            [
                vad_result.get("valence", 0.5),
                vad_result.get("arousal", 0.5),
                vad_result.get("dominance", 0.5),
            ],
        )

        # Extract all sentences from the emotional response
        import json

        try:
            response_data = json.loads(emotional_response)
            sentences = []
            for sentence_key in response_data.get("response", {}):
                sentence_data = response_data["response"][sentence_key]
                sentences.append(sentence_data["text"])

            # Combine all sentences
            combined_text = " ".join(sentences)

            # Generate TTS audio
            tts_result = tts_query({"text": combined_text})

            # Save the audio file with incrementing number if needed
            base_filename = "output.wav"
            counter = 1
            while os.path.exists(base_filename):
                base_filename = f"output({counter}).wav"
                counter += 1

            if (
                "audio" in tts_result
                and isinstance(tts_result["audio"], dict)
                and "url" in tts_result["audio"]
            ):
                audio_url = tts_result["audio"]["url"]
                audio_response = requests.get(audio_url)

                if audio_response.status_code == 200:
                    with open(base_filename, "wb") as f:
                        f.write(audio_response.content)
                    logger.info(f"Saved audio as {base_filename}")
                else:
                    logger.error(f"Failed to download audio from URL: {audio_url}")
            else:
                logger.error("Unexpected TTS response format")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing emotional response: {e}")
            emotional_response = "Error processing emotional response"

        return {
            "filename": file.filename,
            "transcription": transcription_result,
            "vad_scores": vad_result,
            "emotional_response": emotional_response,
            "tts_file": base_filename,
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


@app.get("/healthcheck/")
def healthcheck() -> Dict[str, str]:
    """Health check endpoint to verify server status."""
    logger.info("Health check endpoint accessed.")
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)
