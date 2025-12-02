"""
Therapy Chat Demo (Pilot) - FastAPI Backend
============================================
A supervised, privacy-focused, low-latency GPT therapy tool.

PRIVACY PRINCIPLES:
- No PHI (Protected Health Information) is stored
- No transcripts, messages, or audio are stored
- Only billing metadata (session_id, duration, model, timestamp) is retained
- No clinical content is ever logged
"""

import os
import time
import uuid
import json
import asyncio
import base64
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from openai import OpenAI

app = FastAPI(title="Therapy Chat Demo (Pilot)")
templates = Jinja2Templates(directory="templates")

_client = None

def get_openai_client() -> OpenAI:
    """Get or create OpenAI client. Uses OPENAI_API_KEY from environment."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

MASTER_THERAPY_PROMPT = """
You are an AI reflective tool used within psychotherapy under supervision.
You respond with warmth, clarity, and psychological insight, while strictly avoiding diagnosis,
treatment plans, or risk assessment. You work alongside a licensed therapist, not as a replacement.

IMPORTANT: You MUST always respond in English only, regardless of what language the patient uses.
If the patient speaks in another language, respond in English and gently encourage them to continue in English.

Core Guidelines:
1. Be empathetic, warm, and supportive in your responses
2. Use reflective listening techniques
3. Ask open-ended questions to encourage exploration
4. Validate emotions without making clinical judgments
5. Never provide diagnoses, treatment plans, or medication recommendations
6. Never assess risk or provide crisis intervention
7. Encourage the patient to work with their therapist on deeper issues
8. Keep responses concise but meaningful
9. Focus on the present moment and lived experience
10. Always respond in English only

[TODO: The client's full Master Therapist Instructions will be pasted here.]
"""


@dataclass
class SessionState:
    """
    Tracks session state for a single WebSocket connection.
    Note: Only billing metadata is retained after session ends.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paused: bool = False
    terminated: bool = False
    therapist_whisper: Optional[str] = None
    tone_instruction: Optional[str] = None
    limit_depth: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    conversation_history: List[Dict] = field(default_factory=list)


BILLING_RECORDS: List[Dict] = []


def record_billing(state: SessionState, model_name: str) -> None:
    """
    Record billing metadata for a completed session.
    PRIVACY: No clinical content is stored here - only billing metadata.
    """
    if state.start_time is None or state.end_time is None:
        return
    duration = state.end_time - state.start_time
    BILLING_RECORDS.append({
        "session_id": state.session_id,
        "duration_seconds": round(duration, 2),
        "model": model_name,
        "timestamp": time.time(),
    })
    print(f"Session {state.session_id} completed. Duration: {duration:.2f}s")


import re

def input_safety_gate(text: str) -> tuple[bool, Optional[str]]:
    """
    Input safety gate - blocks crisis content and PHI.
    PRIVACY: This gate prevents sensitive content from being processed.
    Returns (allowed, replacement_message_if_blocked).
    No clinical content is stored - only the gate decision is used.
    """
    lower = text.lower()

    crisis_keywords = [
        "suicide", "kill myself", "end my life", "self-harm",
        "hurt myself", "kill someone", "hurt someone", "homicide",
        "want to die", "ending it all", "take my life", "harm myself"
    ]
    if any(k in lower for k in crisis_keywords):
        return False, (
            "Therapist intervention required. This AI tool cannot respond to "
            "crisis, safety, or harm content. Please speak directly with your "
            "therapist or local emergency services."
        )

    phi_replacement = (
        "I'm not allowed to receive identifying details such as names, "
        "emails, phone numbers, addresses, or ID numbers. You can describe "
        "your experience without those details."
    )
    
    if "@" in text:
        return False, phi_replacement
    
    phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    if re.search(phone_pattern, text):
        return False, phi_replacement
    
    ssn_pattern = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    if re.search(ssn_pattern, text):
        return False, phi_replacement
    
    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count >= 5:
        return False, phi_replacement
    
    name_patterns = [
        r'\bmy name is\b', r'\bmy full name\b', r'\bi am called\b',
        r'\bi live at\b', r'\bmy address is\b', r'\bmy phone\b',
        r'\bmy number is\b', r'\bmy email\b', r'\bmy ssn\b',
        r'\bmy social security\b', r'\bdate of birth\b', r'\bmy dob\b'
    ]
    for pattern in name_patterns:
        if re.search(pattern, lower):
            return False, phi_replacement

    return True, None


def output_safety_gate(text: str) -> str:
    """
    Output safety gate - prevents AI from providing diagnoses, treatment plans, etc.
    PRIVACY: This ensures the AI stays within its therapeutic boundaries.
    """
    lower = text.lower()
    blocked_keywords = [
        "diagnose", "diagnosis", "disorder", "treatment plan",
        "medication", "prescription", "therapy plan",
        "risk assessment", "assess your risk"
    ]
    if any(k in lower for k in blocked_keywords):
        return (
            "I can't diagnose, create treatment plans, or assess risk. "
            "We can stay focused on your present-moment experience, how this "
            "affects you, and possible options to explore with your therapist."
        )
    return text


def build_messages(
    state: SessionState,
    patient_text: str,
) -> List[Dict]:
    """
    Build the message list for OpenAI API calls.
    Always includes the master therapy prompt as the first system message.
    """
    messages: List[Dict] = [
        {"role": "system", "content": MASTER_THERAPY_PROMPT}
    ]
    
    combined_whisper_parts = []
    if state.therapist_whisper:
        combined_whisper_parts.append(state.therapist_whisper)
    if state.tone_instruction:
        combined_whisper_parts.append(f"Tone instruction: {state.tone_instruction}")
    if state.limit_depth:
        combined_whisper_parts.append("Stay surface-level and avoid deep emotional processing.")
    
    if combined_whisper_parts:
        messages.append({"role": "system", "content": " ".join(combined_whisper_parts)})
    
    for msg in state.conversation_history:
        messages.append(msg)
    
    messages.append({"role": "user", "content": patient_text})
    
    return messages


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/billing-debug")
async def billing_debug():
    """
    Debug endpoint to view billing records.
    DEMO ONLY - In production this would require authentication.
    PRIVACY: Only billing metadata is returned - no clinical content.
    """
    return {"billing_records": BILLING_RECORDS, "note": "Only billing metadata is stored - no clinical content."}


@app.post("/api/tts")
async def text_to_speech(request: Request):
    """
    Convert text to speech using OpenAI TTS.
    PRIVACY: Audio is streamed directly - never stored to disk.
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        
        calm_text = (
            "Please speak slowly, calmly, and gently in a therapeutic tone: "
            + text
        )
        
        response = get_openai_client().audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=calm_text,
            response_format="mp3"
        )
        
        audio_content = response.content
        
        return Response(
            content=audio_content,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Convert speech to text using OpenAI Whisper.
    PRIVACY: Audio is processed in memory - never stored to disk.
    """
    try:
        audio_content = await file.read()
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
            tmp.write(audio_content)
            tmp.flush()
            
            with open(tmp.name, "rb") as audio_file:
                transcript = get_openai_client().audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
        
        return {"text": transcript.text}
    except Exception as e:
        print(f"STT error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming text chat.
    PRIVACY: No messages are stored - only billing metadata is retained.
    """
    await websocket.accept()
    state = SessionState()
    model_name = "gpt-4o"
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "user_message":
                text = data.get("text", "").strip()
                if not text:
                    continue
                
                if state.paused:
                    await websocket.send_json({
                        "type": "chunk",
                        "text": "Session is paused by therapist. Please wait."
                    })
                    await websocket.send_json({"type": "final", "text": "Session is paused by therapist. Please wait."})
                    await websocket.send_json({"type": "done"})
                    continue
                
                allowed, replacement = input_safety_gate(text)
                if not allowed:
                    await websocket.send_json({"type": "chunk", "text": replacement})
                    await websocket.send_json({"type": "final", "text": replacement})
                    await websocket.send_json({"type": "done"})
                    continue
                
                if state.start_time is None:
                    state.start_time = time.time()
                
                messages = build_messages(state, text)
                
                try:
                    stream = get_openai_client().chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=True,
                    )
                    
                    assistant_full = ""
                    first_chunk_sent = False
                    for chunk in stream:
                        if state.terminated:
                            break
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            assistant_full += delta
                            if not first_chunk_sent:
                                await websocket.send_json({"type": "chunk", "text": "..."})
                                first_chunk_sent = True
                    
                    filtered_text = output_safety_gate(assistant_full)
                    
                    for i in range(0, len(filtered_text), 20):
                        chunk_text = filtered_text[i:i+20]
                        await websocket.send_json({"type": "chunk", "text": chunk_text})
                        await asyncio.sleep(0.02)
                    
                    await websocket.send_json({"type": "final", "text": filtered_text})
                    await websocket.send_json({"type": "done"})
                    
                    state.conversation_history.append({"role": "user", "content": text})
                    state.conversation_history.append({"role": "assistant", "content": filtered_text})
                    
                    state.end_time = time.time()
                    record_billing(state, model_name)
                    
                except Exception as e:
                    print(f"OpenAI API error: {str(e)}")
                    await websocket.send_json({"type": "error", "text": f"AI error: {str(e)}"})
            
            elif msg_type == "whisper":
                command = data.get("command", "").upper()
                whisper_data = data.get("data", "")
                
                if command == "STOP":
                    state.terminated = True
                elif command == "PAUSE":
                    state.paused = True
                    await websocket.send_json({"type": "status", "message": "Session paused"})
                elif command == "RESUME":
                    state.paused = False
                    await websocket.send_json({"type": "status", "message": "Session resumed"})
                elif command == "TERMINATE":
                    state.terminated = True
                    state.end_time = time.time()
                    record_billing(state, model_name)
                    await websocket.close()
                    return
                elif command == "REDIRECT":
                    state.therapist_whisper = whisper_data
                    await websocket.send_json({"type": "status", "message": "Redirect applied"})
                elif command == "CHANGE_TONE":
                    state.tone_instruction = whisper_data
                    await websocket.send_json({"type": "status", "message": "Tone change applied"})
                elif command == "LIMIT_DEPTH":
                    state.limit_depth = True
                    await websocket.send_json({"type": "status", "message": "Depth limit applied"})
            
            elif msg_type == "therapist_message":
                await websocket.send_json({"type": "status", "message": "Therapist message acknowledged"})
            
            elif msg_type == "therapist_voice":
                try:
                    audio_data = data.get("data", "")
                    if not audio_data:
                        continue
                    
                    audio_bytes = base64.b64decode(audio_data)
                    
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
                        tmp.write(audio_bytes)
                        tmp.flush()
                        
                        with open(tmp.name, "rb") as audio_file:
                            transcript = get_openai_client().audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )
                    
                    transcribed_text = transcript.text
                    await websocket.send_json({
                        "type": "therapist_transcript",
                        "text": transcribed_text
                    })
                except Exception as e:
                    print(f"Therapist voice transcription error: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "text": f"Therapist voice error: {str(e)}"
                    })
    
    except WebSocketDisconnect:
        state.end_time = time.time()
        record_billing(state, model_name)
        print(f"Session {state.session_id} disconnected.")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        state.end_time = time.time()
        record_billing(state, model_name)


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice sessions.
    Uses OpenAI Realtime API for continuous audio in/out.
    PRIVACY: No audio or transcripts are stored - only billing metadata is retained.
    """
    await websocket.accept()
    state = SessionState()
    state.start_time = time.time()
    model_name = "gpt-4o-realtime-preview"
    
    openai_ws = None
    
    try:
        import websockets
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        openai_url = f"wss://api.openai.com/v1/realtime?model={model_name}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        openai_ws = await websockets.connect(openai_url, additional_headers=headers)
        
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": MASTER_THERAPY_PROMPT,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }
        await openai_ws.send(json.dumps(session_update))
        
        assistant_transcript_buffer = ""
        audio_buffer = []
        response_in_progress = False
        
        async def receive_from_openai():
            """Receive messages from OpenAI Realtime API and forward to client."""
            nonlocal assistant_transcript_buffer, audio_buffer, response_in_progress
            try:
                async for message in openai_ws:
                    if state.terminated:
                        break
                    
                    data = json.loads(message)
                    event_type = data.get("type", "")
                    
                    if event_type == "response.created":
                        response_in_progress = True
                        assistant_transcript_buffer = ""
                        audio_buffer = []
                    
                    elif event_type == "response.audio.delta":
                        audio_data = data.get("delta", "")
                        if audio_data and not state.paused:
                            audio_buffer.append(audio_data)
                    
                    elif event_type == "response.audio_transcript.delta":
                        transcript = data.get("delta", "")
                        if transcript:
                            assistant_transcript_buffer += transcript
                    
                    elif event_type == "response.done":
                        if assistant_transcript_buffer:
                            filtered = output_safety_gate(assistant_transcript_buffer)
                            is_blocked = filtered != assistant_transcript_buffer
                            
                            if is_blocked:
                                await websocket.send_json({
                                    "type": "assistant_transcript",
                                    "text": filtered
                                })
                            else:
                                await websocket.send_json({
                                    "type": "assistant_transcript",
                                    "text": filtered
                                })
                                for audio_chunk in audio_buffer:
                                    await websocket.send_json({
                                        "type": "audio",
                                        "data": audio_chunk
                                    })
                        
                        assistant_transcript_buffer = ""
                        audio_buffer = []
                        response_in_progress = False
                    
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = data.get("transcript", "")
                        if transcript:
                            allowed, replacement = input_safety_gate(transcript)
                            if allowed:
                                await websocket.send_json({
                                    "type": "user_transcript",
                                    "text": transcript
                                })
                            else:
                                await websocket.send_json({
                                    "type": "user_transcript",
                                    "text": "[Content filtered]"
                                })
                                await websocket.send_json({
                                    "type": "assistant_transcript",
                                    "text": replacement
                                })
                    
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get("message", "Unknown error")
                        await websocket.send_json({
                            "type": "error",
                            "text": error_msg
                        })
                    
                    elif event_type == "session.created":
                        await websocket.send_json({
                            "type": "status",
                            "message": "Voice session ready"
                        })
                    
            except Exception as e:
                print(f"OpenAI receive error: {str(e)}")
        
        receive_task = asyncio.create_task(receive_from_openai())
        
        try:
            while not state.terminated:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    if not state.paused and openai_ws:
                        audio_data = data.get("data", "")
                        if audio_data:
                            audio_event = {
                                "type": "input_audio_buffer.append",
                                "audio": audio_data
                            }
                            await openai_ws.send(json.dumps(audio_event))
                
                elif msg_type == "whisper":
                    command = data.get("command", "").upper()
                    whisper_data = data.get("data", "")
                    
                    if command == "STOP":
                        state.terminated = True
                    elif command == "PAUSE":
                        state.paused = True
                        await websocket.send_json({"type": "status", "message": "Voice session paused"})
                    elif command == "RESUME":
                        state.paused = False
                        await websocket.send_json({"type": "status", "message": "Voice session resumed"})
                    elif command == "TERMINATE":
                        state.terminated = True
                        break
                    elif command in ["REDIRECT", "CHANGE_TONE", "LIMIT_DEPTH"]:
                        new_instructions = MASTER_THERAPY_PROMPT
                        if command == "REDIRECT":
                            new_instructions += f"\n\nTherapist instruction: {whisper_data}"
                        elif command == "CHANGE_TONE":
                            new_instructions += f"\n\nTone instruction: {whisper_data}"
                        elif command == "LIMIT_DEPTH":
                            new_instructions += "\n\nStay surface-level and avoid deep emotional processing."
                        
                        update_msg = {
                            "type": "session.update",
                            "session": {
                                "instructions": new_instructions
                            }
                        }
                        await openai_ws.send(json.dumps(update_msg))
                        await websocket.send_json({"type": "status", "message": f"{command} applied"})
                
                elif msg_type == "therapist_voice":
                    try:
                        audio_data = data.get("data", "")
                        if not audio_data:
                            continue
                        
                        audio_bytes = base64.b64decode(audio_data)
                        
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
                            tmp.write(audio_bytes)
                            tmp.flush()
                            
                            with open(tmp.name, "rb") as audio_file:
                                transcript = get_openai_client().audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file
                                )
                        
                        transcribed_text = transcript.text
                        await websocket.send_json({
                            "type": "therapist_transcript",
                            "text": transcribed_text
                        })
                    except Exception as e:
                        print(f"Therapist voice transcription error: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "text": f"Therapist voice error: {str(e)}"
                        })
        
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
    
    except WebSocketDisconnect:
        print(f"Voice session {state.session_id} disconnected.")
    except Exception as e:
        print(f"Voice WebSocket error: {str(e)}")
        await websocket.send_json({"type": "error", "text": str(e)})
    finally:
        state.end_time = time.time()
        record_billing(state, model_name)
        
        if openai_ws:
            try:
                await openai_ws.close()
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
