
# backend.py
import asyncio
import json
import base64
import io
import os
import wave
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import transformers
import librosa
import torch
from smallestai.waves import WavesClient  # Correct import from documentation
import tempfile
from typing import AsyncGenerator
import logging
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SMALLEST_API_KEY = os.getenv("SMALLEST_API_KEY", "your-smallest-api-key")

# Initialize Ultravox model (downloads from HuggingFace on first run)
logger.info("Loading Ultravox model from HuggingFace...")
ultravox_pipe = transformers.pipeline(
    model='fixie-ai/ultravox-v0_5-llama-3_1-8b',
    trust_remote_code=True,
    device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
logger.info("Ultravox model loaded successfully")

# Initialize SmallestAI TTS client
tts_client = WavesClient(api_key=SMALLEST_API_KEY)

# Custom TextToAudioStream implementation based on documentation
class TextToAudioStream:
    """Custom implementation of TextToAudioStream for real-time TTS"""
    
    def __init__(self, tts_instance, queue_timeout=5.0, max_retries=3):
        self.tts = tts_instance
        self.queue_timeout = queue_timeout
        self.max_retries = max_retries
    
    async def process(self, text_generator):
        """Process text stream and yield audio chunks"""
        text_buffer = ""
        sentence_delimiters = ['.', '!', '?', '\n']
        
        async for text_chunk in text_generator:
            if text_chunk:
                text_buffer += text_chunk
                
                # Check if we have a complete sentence
                for delimiter in sentence_delimiters:
                    if delimiter in text_buffer:
                        sentences = text_buffer.split(delimiter)
                        
                        # Process complete sentences
                        for sentence in sentences[:-1]:
                            sentence = sentence.strip()
                            if sentence:
                                # Generate audio for this sentence
                                for retry in range(self.max_retries):
                                    try:
                                        # Synthesize audio
                                        audio_data = self.tts.synthesize(
                                            sentence + delimiter,
                                            save_as=None  # Return bytes instead of saving
                                        )
                                        
                                        if audio_data:
                                            # Yield audio chunks
                                            chunk_size = 4096
                                            for i in range(0, len(audio_data), chunk_size):
                                                yield audio_data[i:i + chunk_size]
                                        break
                                    except Exception as e:
                                        logger.error(f"TTS error (retry {retry + 1}): {e}")
                                        if retry == self.max_retries - 1:
                                            logger.error(f"Failed to synthesize: {sentence}")
                        
                        # Keep the incomplete sentence in buffer
                        text_buffer = sentences[-1]
        
        # Process any remaining text
        if text_buffer.strip():
            try:
                audio_data = self.tts.synthesize(text_buffer.strip(), save_as=None)
                if audio_data:
                    chunk_size = 4096
                    for i in range(0, len(audio_data), chunk_size):
                        yield audio_data[i:i + chunk_size]
            except Exception as e:
                logger.error(f"TTS error for final text: {e}")

class VoiceAgent:
    def __init__(self):
        self.conversation_history = []
        
    async def process_audio_with_ultravox(self, audio_data: bytes) -> str:
        """
        Process audio using Ultravox which handles both:
        1. Speech-to-text (understanding the audio)
        2. Response generation (acting as an LLM)
        """
        try:
            # Convert base64 to bytes if needed
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)
            
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
                
                # Convert webm/opus to WAV (since browser sends webm)
                try:
                    # Try to load audio data with pydub
                    audio_segment = AudioSegment.from_file(BytesIO(audio_data))
                    # Convert to mono, 16kHz, 16-bit WAV
                    audio_segment = audio_segment.set_channels(1)
                    audio_segment = audio_segment.set_frame_rate(16000)
                    audio_segment = audio_segment.set_sample_width(2)
                    # Export as WAV
                    audio_segment.export(temp_path, format="wav")
                except Exception as e:
                    logger.warning(f"Pydub conversion failed, trying direct write: {e}")
                    # If pydub fails, assume it's already WAV
                    with open(temp_path, 'wb') as f:
                        f.write(audio_data)
            
            # Load audio using librosa
            audio, sr = librosa.load(temp_path, sr=16000)
            
            # Prepare conversation turns for Ultravox
            turns = [
                {
                    "role": "system",
                    "content": "You are a friendly and helpful AI assistant. Keep your responses concise and natural for voice conversation. Be conversational and engaging."
                }
            ]
            
            # Add recent conversation history (keep context manageable)
            for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                turns.append(msg)
            
            # Process with Ultravox - it will understand the audio and generate a response
            logger.info("Processing with Ultravox...")
            result = ultravox_pipe(
                {
                    'audio': audio,
                    'turns': turns,
                    'sampling_rate': sr
                },
                max_new_tokens=100  # Adjust for response length
            )
            
            # Extract the response text
            response_text = result
            if isinstance(result, dict):
                response_text = result.get('generated_text', '') or result.get('text', '')
            
            logger.info(f"Ultravox response: {response_text[:100]}...")
            
            # Update conversation history with the assistant's response
            self.conversation_history.append({"role": "user", "content": "[Voice input received]"})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing audio with Ultravox: {e}", exc_info=True)
            return "I'm sorry, I had trouble understanding that. Could you please try again?"
    
    async def text_to_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream text to speech using SmallestAI"""
        try:
            # Create async generator for text
            async def text_generator():
                # Split text into smaller chunks for streaming
                sentences = text.replace('!', '.').replace('?', '.').split('.')
                for sentence in sentences:
                    if sentence.strip():
                        yield sentence.strip() + '. '
                        await asyncio.sleep(0.01)  # Small delay for natural pacing
            
            # Use TextToAudioStream for real-time TTS
            processor = TextToAudioStream(tts_instance=tts_client)
            
            logger.info(f"Starting TTS streaming for: {text[:50]}...")
            chunk_count = 0
            
            async for audio_chunk in processor.process(text_generator()):
                chunk_count += 1
                yield audio_chunk
            
            logger.info(f"TTS streaming complete. Sent {chunk_count} chunks")
                
        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}", exc_info=True)
            # Return silence on error
            yield b'\x00' * 1024
    
    async def text_to_speech_simple(self, text: str) -> bytes:
        """Simple non-streaming TTS for complete audio generation"""
        try:
            logger.info(f"Generating speech for: {text[:50]}...")
            
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Synthesize and save to file
            tts_client.synthesize(text, save_as=temp_path)
            
            # Read the audio file
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            logger.info(f"Generated {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in TTS: {e}", exc_info=True)
            # Return empty WAV on error
            return self._create_empty_wav()
    
    def _create_empty_wav(self) -> bytes:
        """Create an empty WAV file as fallback"""
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b'\x00' * 1024)
        return buffer.getvalue()

# Store agent instances per connection
agents = {}

@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice conversation"""
    await websocket.accept()
    
    # Create a unique agent for this connection
    connection_id = id(websocket)
    agents[connection_id] = VoiceAgent()
    
    try:
        logger.info(f"New WebSocket connection established: {connection_id}")
        
        # Send initial connection confirmation
        await websocket.send_json({
            'type': 'connection_established',
            'message': 'Ready for voice input'
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get('type') == 'audio':
                # Extract audio data
                audio_base64 = data.get('audio', '')
                
                if not audio_base64:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'No audio data received'
                    })
                    continue
                
                logger.info("Received audio input, processing with Ultravox...")
                
                # Process audio with Ultravox (STT + Response Generation in one step)
                response_text = await agents[connection_id].process_audio_with_ultravox(audio_base64)
                
                # Send the text response back to client
                await websocket.send_json({
                    'type': 'response_text',
                    'text': response_text
                })
                
                # Choose between streaming or simple TTS
                use_streaming = True  # Set to False for simple mode
                
                if use_streaming:
                    # Stream audio in chunks for real-time feel
                    logger.info("Streaming audio response...")
                    audio_chunks_sent = 0
                    
                    async for audio_chunk in agents[connection_id].text_to_speech_stream(response_text):
                        # Convert audio chunk to base64 for transmission
                        audio_base64_chunk = base64.b64encode(audio_chunk).decode('utf-8')
                        
                        await websocket.send_json({
                            'type': 'audio_chunk',
                            'audio': audio_base64_chunk,
                            'chunk_index': audio_chunks_sent
                        })
                        audio_chunks_sent += 1
                        
                        # Small delay to prevent overwhelming the client
                        if audio_chunks_sent % 10 == 0:
                            await asyncio.sleep(0.01)
                    
                    # Signal end of audio stream
                    await websocket.send_json({
                        'type': 'audio_complete',
                        'total_chunks': audio_chunks_sent
                    })
                    
                    logger.info(f"Streamed {audio_chunks_sent} audio chunks")
                else:
                    # Send complete audio at once
                    logger.info("Generating complete audio response...")
                    audio_data = await agents[connection_id].text_to_speech_simple(response_text)
                    
                    # Send as single chunk
                    audio_base64_response = base64.b64encode(audio_data).decode('utf-8')
                    await websocket.send_json({
                        'type': 'audio_chunk',
                        'audio': audio_base64_response,
                        'chunk_index': 0
                    })
                    
                    await websocket.send_json({
                        'type': 'audio_complete',
                        'total_chunks': 1
                    })
                    
                    logger.info("Sent complete audio response")
                
            elif data.get('type') == 'ping':
                # Handle keepalive pings
                await websocket.send_json({'type': 'pong'})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                'type': 'error',
                'message': 'An error occurred processing your request'
            })
        except:
            pass
    finally:
        # Clean up agent instance
        if connection_id in agents:
            del agents[connection_id]
            logger.info(f"Cleaned up agent for connection: {connection_id}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/")
async def root():
    return {
        "message": "Real-time Voice Agent API",
        "status": "running",
        "model": "Ultravox (Speech-to-Response)",
        "tts": "SmallestAI",
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ultravox": "loaded" if ultravox_pipe else "not loaded",
        "tts": "ready",
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Print startup information
    print("=" * 50)
    print("Real-time Voice Agent Server")
    print("=" * 50)
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"SmallestAI API Key: {'Set' if SMALLEST_API_KEY != 'your-smallest-api-key' else 'Not Set'}")
    print("Starting server on http://localhost:8000")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")