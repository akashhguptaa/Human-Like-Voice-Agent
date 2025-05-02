from loguru import logger
import soundfile as sf
import asyncio
import librosa
import io
import numpy as np



async def load_and_preprocess_audio(file_content: bytes, target_sr: int) -> np.ndarray:
    """
    Loads audio from bytes, converts to mono, resamples, and ensures float32 format.
    Returns a NumPy array. (Identical to previous version)
    """
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
