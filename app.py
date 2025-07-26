import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError
import logging
from datetime import datetime

# --- Configuration ---
COMPUTE_TYPE = "float16" # Changed to float16 for better cuda compatibility
BATCH_SIZE = 16 # Reduced batch size for cuda
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console
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
        # Test if directory is writable
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def convert_to_wav(input_path: str) -> str:
    """Convert media file to 16kHz mono WAV with proper timestamps"""
    try:
        output_path = f"/tmp/{uuid.uuid4()}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-af", "aresample=async=1:first_pts=0",
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
    """Load Whisper model with language-specific optimizations"""
    try:
        if not ensure_model_cache_dir():
            raise RuntimeError("Model cache directory is not accessible")
            
        # Language-specific parameters
        params = {
            "device": "cuda",
            "compute_type": COMPUTE_TYPE,
            "download_root": MODEL_CACHE_DIR,
            "language": language if language and language != "-" else None
        }
        
        # Special handling for Indian languages
        if language in ['gu', 'hi', 'ta', 'mr', 'bn', 'pa', 'te', 'kn', 'ml', 'or']:
            params["temperature"] = 0.2
            params["best_of"] = 5
            params["beam_size"] = 5
            
        return whisperx.load_model(model_size, **params)
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription logic with improved alignment for all languages"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language else "en")
        
        if align and detected_language != "unknown":
            try:
                # First try default alignment
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device="cuda"
                    )
                except (ValueError, RuntimeError) as e:
                    # Custom model mapping for better language support
                    custom_models = {
                        'gu': 'ai4bharat/indicwav2vec_v1_gujarati',
                        'hi': 'ai4bharat/indicwav2vec_v1_hindi',
                        'ta': 'ai4bharat/indicwav2vec_v1_tamil',
                        'mr': 'ai4bharat/indicwav2vec_v1_marathi',
                        'bn': 'ai4bharat/indicwav2vec_v1_bengali',
                        'pa': 'ai4bharat/indicwav2vec_v1_punjabi',
                        'te': 'ai4bharat/indicwav2vec_v1_telugu',
                        'kn': 'ai4bharat/indicwav2vec_v1_kannada',
                        'ml': 'ai4bharat/indicwav2vec_v1_malayalam',
                        'or': 'ai4bharat/indicwav2vec_v1_odia'
                    }
                    
                    if detected_language in custom_models:
                        logger.info(f"Using custom alignment model for {detected_language}")
                        align_model, metadata = whisperx.load_align_model(
                            model_name=custom_models[detected_language],
                            device="cuda"
                        )
                    else:
                        raise ValueError(f"No suitable alignment model for {detected_language}")

                # Perform alignment with improved parameters
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device="cuda",
                    return_char_alignments=False,
                    interpolate_method="nearest",
                    output_word_timestamps=True
                )
                
                # Ensure all segments have word-level data
                for seg in result["segments"]:
                    if "words" not in seg or not seg["words"]:
                        seg["words"] = [{
                            "word": seg["text"],
                            "start": seg["start"],
                            "end": seg["end"],
                            "score": 0.9
                        }]
                        
            except Exception as e:
                logger.warning(f"Alignment processing for {detected_language}: {str(e)}")
                # Fallback: Create basic word-level data
                for seg in result["segments"]:
                    seg["words"] = [{
                        "word": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "score": 0.9
                    }]

        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "segments": result["segments"],
            "language": detected_language,
            "model": model_size,
            "alignment_skipped": align and detected_language != "unknown" and "words" not in result["segments"][0]
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def handler(job):
    """RunPod serverless handler with improved error handling"""
    try:
        # Validate input
        if not job.get("input"):
            return {"error": "No input provided", "failed": True}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        
        if not file_name:
            return {"error": "No file_name provided in input", "failed": True}
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
            logger.info(f"Downloaded file from S3: {file_name}")
        except Exception as e:
            error_msg = f"S3 download failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "failed": True}
        
        # 2. Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                logger.info("Converting audio to WAV format")
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                audio_path = local_path
        except Exception as e:
            error_msg = f"Audio processing failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "failed": True}
        
        # 3. Transcribe with language detection
        try:
            logger.info(f"Starting transcription with model: {input_data.get('model_size', 'large-v3')}")
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language"),
                input_data.get("align", False)
            )
            logger.info(f"Transcription completed for language: {result['language']}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Transcription error: {error_msg}")
            return {"error": error_msg, "failed": True}
        finally:
            # 4. Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.critical(error_msg)
        return {"error": error_msg, "failed": True}

if __name__ == "__main__":
    print("Starting Enhanced WhisperX Endpoint...")
    
    if not ensure_model_cache_dir():
        error_msg = "Model cache directory is not accessible"
        print(f"ERROR: {error_msg}")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            raise RuntimeError(error_msg)
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        test_result = handler({
            "input": {
                "file_name": "gujarati_audio.mp3",
                "model_size": "large-v3",
                "align": True
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
