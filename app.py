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
import logging
from datetime import datetime

# --- Configuration ---
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")
DEFAULT_MAX_WORDS = 5  # Default max words per segment

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
        logger.error(f"FFmpeg conversion failed: {str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    try:
        if not ensure_model_cache_dir():
            raise RuntimeError("Model cache directory is not accessible")
            
        return whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def split_segments_by_word_count(segments, max_words):
    """Split segments by exact word count while preserving timestamps"""
    if max_words <= 0:
        return segments
        
    new_segments = []
    for seg in segments:
        words = seg['text'].strip().split()
        word_count = len(words)
        
        if word_count <= max_words:
            new_segments.append(seg)
            continue
            
        # Split into chunks of max_words
        for i in range(0, word_count, max_words):
            chunk_words = words[i:i+max_words]
            chunk_text = ' '.join(chunk_words)
            
            new_segment = {
                'text': chunk_text,
                'start': seg['start'],
                'end': seg['end'],
                # Copy all other fields
                **{k: v for k, v in seg.items() if k not in ['text', 'start', 'end', 'words']}
            }
            
            # Handle word-level timestamps if available
            if 'words' in seg and isinstance(seg['words'], list) and len(seg['words']) > 0:
                start_idx = min(i, len(seg['words'])-1)
                end_idx = min(i+max_words-1, len(seg['words'])-1)
                new_segment['start'] = seg['words'][start_idx]['start']
                new_segment['end'] = seg['words'][end_idx]['end']
                new_segment['words'] = seg['words'][start_idx:end_idx+1]
                
            new_segments.append(new_segment)
    return new_segments

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool, max_words: int = DEFAULT_MAX_WORDS):
    """Core transcription logic with word-level segmentation"""
    try:
        model = load_model(model_size, language)
        
        # Get initial transcription
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        detected_language = result.get("language", language if language else "en")
        
        # Apply alignment if requested
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
        
        # Apply word-level segmentation
        if max_words > 0:
            result['segments'] = split_segments_by_word_count(result['segments'], max_words)
        
        return {
            'text': ' '.join(seg['text'] for seg in result['segments']),
            'segments': result['segments'],
            'language': detected_language,
            'model': model_size,
            'max_words_per_segment': max_words,
            'segment_count': len(result['segments'])
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
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
        
        # Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            if S3_BUCKET:
                s3.download_file(S3_BUCKET, file_name, local_path)
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}
        
        # Convert to WAV if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                audio_path = local_path
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}
        
        # Transcribe with word-level control
        try:
            result = transcribe_audio(
                audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language", None),
                input_data.get("align", False),
                max_words=int(input_data.get("max_words", DEFAULT_MAX_WORDS))
        except Exception as e:
            return {"error": str(e)}
        finally:
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()
        
        return result
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    print("Starting WhisperX Endpoint with Word-Level Segmentation...")
    
    if not ensure_model_cache_dir():
        print("ERROR: Model cache directory is not accessible")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            raise RuntimeError("Model cache directory is not accessible")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input
        test_result = handler({
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "align": True,
                "max_words": 3  # Test with strict 3-word segments
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
