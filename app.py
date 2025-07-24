import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
import time
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError
import logging
from datetime import datetime
from deep_translator import GoogleTranslator
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Configuration ---
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")
MAX_TRANSLATION_LENGTH = 5000  # Google Translate character limit
TRANSLATION_RETRY_ATTEMPTS = 3

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

def ensure_model_cache_dir():
    """Ensure model cache directory exists and is accessible"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV"""
    try:
        output_path = f"/tmp/{uuid.uuid4()}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg conversion failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Audio conversion error: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    try:
        if not ensure_model_cache_dir():
            error_msg = "Model cache directory is not accessible"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        return whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

@retry(
    stop=stop_after_attempt(TRANSLATION_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def safe_translate(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Robust translation with retry logic"""
    if len(text) > MAX_TRANSLATION_LENGTH:
        raise ValueError(f"Text too long for translation ({len(text)} > {MAX_TRANSLATION_LENGTH} chars)")
    
    return GoogleTranslator(
        source=source_lang,
        target=target_lang
    ).translate(text)

def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Handle translation with chunking and error management"""
    if not text or target_lang == "-":
        return text
    
    try:
        # Split long text into chunks if needed
        if len(text) > MAX_TRANSLATION_LENGTH:
            chunks = [text[i:i+MAX_TRANSLATION_LENGTH] for i in range(0, len(text), MAX_TRANSLATION_LENGTH)]
            translated_chunks = []
            for chunk in chunks:
                translated = safe_translate(chunk, target_lang, source_lang)
                translated_chunks.append(translated)
                time.sleep(0.5)  # Rate limit between chunks
            return " ".join(translated_chunks)
        
        return safe_translate(text, target_lang, source_lang)
    except Exception as e:
        error_msg = f"Translation failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def transcribe_audio(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    align: bool,
    translate_to: Optional[str] = None
) -> Dict[str, Any]:
    """Core transcription logic with strict error handling"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language else "en")
        
        if align and detected_language != "unknown":
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device="cuda"
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device="cuda",
                    return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment skipped: {str(e)}")

        full_text = " ".join(seg["text"] for seg in result["segments"])
        output = {
            "language": detected_language,
            "model": model_size,
            "segments": result["segments"],
            "text": full_text
        }

        # Only add translations if requested
        if translate_to and translate_to != "-":
            try:
                output["translated_text"] = translate_text(full_text, translate_to, detected_language)
                output["translation_target"] = translate_to
                
                for segment in result["segments"]:
                    segment["textTranslated"] = translate_text(segment["text"], translate_to, detected_language)
                    
                    if "words" in segment:
                        for word in segment["words"]:
                            word["wordTranslated"] = translate_text(word["word"], translate_to, detected_language)
            except Exception as e:
                # If translation fails at any point, fail the entire operation
                raise RuntimeError(f"Translation processing failed: {str(e)}")

        return output

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def handler(job):
    """RunPod serverless handler with strict error handling"""
    try:
        if not job.get("input"):
            return {"error": "No input provided"}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        
        if not file_name:
            return {"error": "No file_name provided in input"}
        
        # Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
        except Exception as e:
            error_msg = f"S3 download failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "failed": True}
        
        # Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                audio_path = local_path
        except Exception as e:
            error_msg = f"Audio processing failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "failed": True}
        
        # Transcribe with strict error handling
        try:
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language"),
                input_data.get("align", False),
                input_data.get("translate")
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(error_msg)
            return {"error": error_msg, "failed": True}
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.critical(error_msg)
        return {"error": error_msg, "failed": True}

if __name__ == "__main__":
    print("Starting WhisperX Endpoint with Strict Error Handling...")
    
    if not ensure_model_cache_dir():
        error_msg = "Model cache directory is not accessible"
        print(f"ERROR: {error_msg}")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            raise RuntimeError(error_msg)
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input
        test_result = handler({
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "align": True,
                "translate": "ar"
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
