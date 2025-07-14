import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
from typing import Optional
from botocore.exceptions import ClientError

# --- Configuration ---
COMPUTE_TYPE = "float16"  # Changed to float16 for better cuda compatibility
BATCH_SIZE = 16  # Reduced batch size for cuda
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")

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
        print(f"Model cache directory error: {str(e)}")
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
        log(f"Fmpeg conversion failed error: {str(e)}", "ERROR")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        log(f"Audio conversion error: {str(e)}", "ERROR")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    try:
        if not ensure_model_cache_dir():
            log(f"Model cache directory is not accessible", "ERROR")
            raise RuntimeError("Model cache directory is not accessible")
            
        return whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
    except Exception as e:
        log(f"Model loading failed: {str(e)}", "ERROR")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription logic"""
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
                log(f"Alignment skipped: {str(e)}", "ERROR")
                print(f"Alignment skipped: {str(e)}")
        
        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "segments": result["segments"],
            "language": detected_language,
            "model": model_size
        }
    except Exception as e:
        log(f"Transcription failed: {str(e)}", "ERROR")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def handler(job):
    """RunPod serverless handler"""
    try:
        # Validate input
        if not job.get("input"):
            return {"error": "No input provided"}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        
        if not file_name:
            return {"error": "No file_name provided in input"}
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}
        
        # 2. Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                audio_path = local_path
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}
        
        # 3. Transcribe
        try:
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language", None),
                input_data.get("align", False)
            )
        except Exception as e:
            return {"error": str(e)}
        finally:
            # 4. Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()
        
        return result
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    print("Starting WhisperX cuda Endpoint...")
    
    # Verify model cache directory at startup
    if not ensure_model_cache_dir():
        print("ERROR: Model cache directory is not accessible")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            # In serverless mode, we want to fail fast if model dir isn't available
            raise RuntimeError("Model cache directory is not accessible")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input
        test_result = handler({
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "align": True
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
