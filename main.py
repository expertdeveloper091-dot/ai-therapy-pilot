import asyncio
import base64
import json
import time
import os 
from typing import Optional

from fastapi import FastAPI, WebSocket, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from openai import OpenAI
# -------------------------------------------------------
# OpenAI CLIENT (FIXED — NO PROXIES, NEW 2025 FORMAT)
# -------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# FASTAPI APP SETUP
# -------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# BASIC HTML HOMEPAGE
# -------------------------------------------------------
@app.get("/")
async def home():
    return {"status": "AI Therapy Pilot is running on Render.com"}
# -------------------------------------------------------
# THERAPIST SYSTEM PROMPT (Main personality & behavior)
# -------------------------------------------------------

THERAPIST_SYSTEM_PROMPT = """
You are a warm, empathetic, and supportive mental health conversation partner.
You help users explore their emotions safely.

Rules:
- Never give medical, legal, or diagnostic claims.
- Never tell users you are a therapist or licensed.
- Encourage grounding, reflection, and safe coping.
- If user expresses self-harm, follow safety_redirect().
"""

# -------------------------------------------------------
# SAFETY FILTERS
# -------------------------------------------------------

def is_self_harm(text: str) -> bool:
    if not text:
        return False
    text = text.lower()

    keywords = [
        "suicide", "kill myself", "end my life",
        "i want to die", "hurt myself",
        "can't live", "life is pointless",
        "self harm", "cut myself"
    ]

    return any(k in text for k in keywords)


def safety_redirect() -> str:
    return (
        "I'm really glad you reached out. You’re not alone, and your feelings matter. "
        "I’m not able to help in crisis situations, but you deserve immediate care from real people who can support you.\n\n"
        "📞 **If you’re in immediate danger, please contact local emergency services.**\n\n"
        "If you can, please also reach out to someone:\n"
        "- Call your local suicide hotline\n"
        "- Contact someone you trust\n"
        "- If available, use your country's crisis helpline\n\n"
        "You don’t have to face this alone."
    )

# -------------------------------------------------------
# FORMAT CHAT MESSAGES FOR MODELS
# -------------------------------------------------------

def build_messages(user_message: str, conversation: list):
    """
    Prepares messages for gpt-4.1, gpt-5.1, gpt-4o-mini (normal chat models).
    """

    messages = [{"role": "system", "content": THERAPIST_SYSTEM_PROMPT}]

    for m in conversation:
        messages.append({"role": m["role"], "content": m["content"]})

    # append latest user message
    messages.append({"role": "user", "content": user_message})
    return messages


# -------------------------------------------------------
# AUDIO DECODING / ENCODING HELPERS
# -------------------------------------------------------

def decode_audio(base64_str: str) -> bytes:
    return base64.b64decode(base64_str)


def encode_audio(binary: bytes) -> str:
    return base64.b64encode(binary).decode()


# -------------------------------------------------------
# TIMESTAMPED LOG UTILITY
# -------------------------------------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
# -------------------------------------------------------
# SPEECH TO TEXT (STT) - Whisper (New OpenAI API 2025)
# -------------------------------------------------------

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """Converts audio file to text using Whisper (OpenAI's /audio/transcriptions)."""

    audio_bytes = await file.read()

    try:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",       # 2025 whisper replacement
            file=("audio.wav", audio_bytes)
        )

        return {"text": response.text}

    except Exception as e:
        log(f"STT Error: {e}")
        return {"error": str(e)}


# -------------------------------------------------------
# TEXT TO SPEECH (TTS) - New 2025 models
# -------------------------------------------------------

@app.post("/tts")
async def tts(request: Request):
    """Converts text → audio using OpenAI TTS models."""

    body = await request.json()
    text = body.get("text", "")
    voice = body.get("voice", "alloy")     # Default voice

    if not text:
        return {"error": "No text provided."}

    try:
        # TTS API (2025)
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",      # lightweight & fast
            voice=voice,
            input=text,
            format="wav"
        )

        audio_bytes = response.read()     # Get binary audio

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav"
        )

    except Exception as e:
        log(f"TTS Error: {e}")
        return {"error": str(e)}
# -------------------------------------------------------
# TEXT CHAT WEBSOCKET (UPDATED FOR OPENAI 2025 API)
# -------------------------------------------------------

@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket):
    await ws.accept()

    state = SessionState()
    model_name = "gpt-5.1"   # default LLM

    try:
        while True:
            data = await ws.receive_json()
            mtype = data.get("type")

            # ---------------------------------------------------
            # USER SENDS MESSAGE
            # ---------------------------------------------------
            if mtype == "user_message":
                text = data.get("text", "").strip()

                if not text:
                    continue

                # Safety gate
                allowed, safe_msg = input_safety_gate(text)
                if not allowed:
                    await ws.send_json({"type": "chunk", "text": safe_msg})
                    await ws.send_json({"type": "final", "text": safe_msg})
                    continue

                # Update model if provided
                requested = data.get("model")
                if requested in VALID_MODELS:
                    model_name = requested

                # Start billing
                if state.start_time is None:
                    state.start_time = time.time()

                # Build message history
                messages = build_messages(state, text)

                # ---------------------------------------------------
                # NEW OPENAI 2025 STREAMING CHAT COMPLETION
                # ---------------------------------------------------

                try:
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=True
                    )

                    collected = ""
                    first = True

                    for event in stream:
                        if state.terminated:
                            break

                        delta = event.choices[0].delta.get("content", "")
                        if not delta:
                            continue

                        collected += delta

                        # send initial "..." indicator
                        if first:
                            await ws.send_json({"type": "chunk", "text": "..."})
                            first = False

                    # Filter unsafe output
                    filtered = output_safety_gate(collected)

                    # send in chunks (smooth streaming effect)
                    for i in range(0, len(filtered), 20):
                        await ws.send_json({
                            "type": "chunk",
                            "text": filtered[i:i+20]
                        })
                        await asyncio.sleep(0.02)

                    await ws.send_json({"type": "final", "text": filtered})

                    # update history
                    state.conversation_history.append({"role": "user", "content": text})
                    state.conversation_history.append({"role": "assistant", "content": filtered})

                    # update billing
                    state.end_time = time.time()
                    record_billing(state, model_name)

                except Exception as e:
                    await ws.send_json({"type": "error", "text": str(e)})

            # ---------------------------------------------------
            # THERAPIST CONTROLS
            # ---------------------------------------------------
            elif mtype == "whisper":
                command = data.get("command", "").upper()
                payload = data.get("data", "")

                if command == "PAUSE":
                    state.paused = True
                    await ws.send_json({"type": "status", "message": "Paused"})

                elif command == "RESUME":
                    state.paused = False
                    await ws.send_json({"type": "status", "message": "Resumed"})

                elif command == "STOP":
                    state.terminated = True

                elif command == "TERMINATE":
                    state.terminated = True
                    state.end_time = time.time()
                    record_billing(state, model_name)
                    await ws.close()
                    return

                elif command == "REDIRECT":
                    state.therapist_whisper = payload
                    await ws.send_json({"type": "status", "message": "Redirect applied"})

                elif command == "CHANGE_TONE":
                    state.tone_instruction = payload
                    await ws.send_json({"type": "status", "message": "Tone updated"})

                elif command == "LIMIT_DEPTH":
                    state.limit_depth = True
                    await ws.send_json({"type": "status", "message": "Depth limit enabled"})

            # ---------------------------------------------------
            # THERAPIST TEXT MESSAGE
            # ---------------------------------------------------
            elif mtype == "therapist_message":
                await ws.send_json({"type": "status", "message": "Therapist note received"})

            # ---------------------------------------------------
            # THERAPIST VOICE → STT
            # ---------------------------------------------------
            elif mtype == "therapist_voice":
                try:
                    b64 = data.get("data", "")
                    if not b64:
                        continue

                    audio_bytes = base64.b64decode(b64)

                    resp = client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=("audio.webm", audio_bytes)
                    )

                    await ws.send_json({
                        "type": "therapist_transcript",
                        "text": resp.text
                    })

                except Exception as e:
                    await ws.send_json({"type": "error", "text": str(e)})

    except WebSocketDisconnect:
        state.end_time = time.time()
        record_billing(state, model_name)
# -------------------------------------------------------
# REAL-TIME VOICE WEBSOCKET  (OpenAI Realtime API 2025)
# -------------------------------------------------------

import websockets

@app.websocket("/ws/voice")
async def voice_socket(ws: WebSocket):
    await ws.accept()

    state = SessionState()
    state.start_time = time.time()

    model_name = "gpt-4o-realtime-preview"

    # connect to OpenAI realtime WS
    api_key = os.environ.get("OPENAI_API_KEY")

    headers = [
        ("Authorization", f"Bearer {api_key}"),
        ("OpenAI-Beta", "realtime=v1")
    ]

    openai_ws = await websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={model_name}",
        extra_headers=headers,
    )

    # -------------------------------------------------------
    # initial session settings
    # -------------------------------------------------------
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": "alloy",
            "instructions": MASTER_THERAPY_PROMPT,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            }
        }
    }

    await openai_ws.send(json.dumps(session_update))

    # -------------------------------------------------------
    # TASK: LISTEN TO OPENAI REALTIME EVENTS
    # -------------------------------------------------------

    async def from_openai():
        """Receive OpenAI realtime events and forward to browser."""
        audio_collected = []
        transcript_collected = ""

        try:
            async for message in openai_ws:
                if state.terminated:
                    break

                evt = json.loads(message)
                etype = evt.get("type", "")

                # ---- assistant started speaking ----
                if etype == "response.created":
                    audio_collected = []
                    transcript_collected = ""

                # ---- text transcript delta ----
                elif etype == "response.audio_transcript.delta":
                    t = evt.get("delta", "")
                    transcript_collected += t

                # ---- audio delta ----
                elif etype == "response.audio.delta":
                    audio_chunk = evt.get("delta", "")
                    if audio_chunk and not state.paused:
                        audio_collected.append(audio_chunk)

                # ---- assistant finished speaking ----
                elif etype == "response.done":
                    # apply safety
                    safe_text = output_safety_gate(transcript_collected)

                    await ws.send_json({
                        "type": "assistant_transcript",
                        "text": safe_text
                    })

                    # send audio chunks
                    for chunk in audio_collected:
                        await ws.send_json({
                            "type": "audio",
                            "data": chunk
                        })

                # ---- user speech transcription ----
                elif etype == "conversation.item.input_audio_transcription.completed":
                    t = evt.get("transcript", "")
                    allowed, safe_msg = input_safety_gate(t)

                    if allowed:
                        await ws.send_json({"type": "user_transcript", "text": t})
                    else:
                        await ws.send_json({"type": "assistant_transcript", "text": safe_msg})

                # ---- errors ----
                elif etype == "error":
                    msg = evt.get("error", {}).get("message", "Error")
                    await ws.send_json({"type": "error", "text": msg})

                # ready
                elif etype == "session.created":
                    await ws.send_json({"type": "status", "message": "Voice session ready"})

        except Exception as e:
            print("Realtime receive error:", e)

    # background listener
    task = asyncio.create_task(from_openai())

    # -------------------------------------------------------
    # BROWSER → OPENAI REALTIME PIPE
    # -------------------------------------------------------

    try:
        while True:
            data = await ws.receive_json()
            mtype = data.get("type")

            # incoming audio
            if mtype == "audio" and not state.paused:
                pcm = data.get("data", "")
                if pcm:
                    await openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": pcm
                    }))

            # therapist commands
            elif mtype == "whisper":
                cmd = data.get("command", "").upper()
                arg = data.get("data", "")

                if cmd == "PAUSE":
                    state.paused = True
                    await ws.send_json({"type": "status", "message": "Paused"})

                elif cmd == "RESUME":
                    state.paused = False
                    await ws.send_json({"type": "status", "message": "Resumed"})

                elif cmd == "STOP":
                    state.terminated = True

                elif cmd == "TERMINATE":
                    state.terminated = True
                    break

                elif cmd in ["REDIRECT", "CHANGE_TONE", "LIMIT_DEPTH"]:
                    new_inst = MASTER_THERAPY_PROMPT

                    if cmd == "REDIRECT":
                        new_inst += f"\nTherapist instruction: {arg}"
                    if cmd == "CHANGE_TONE":
                        new_inst += f"\nTone instruction: {arg}"
                    if cmd == "LIMIT_DEPTH":
                        new_inst += "\nStay surface-level and avoid deep processing."

                    await openai_ws.send(json.dumps({
                        "type": "session.update",
                        "session": {"instructions": new_inst}
                    }))

                    await ws.send_json({"type": "status", "message": f"{cmd} applied"})

            # therapist voice → STT
            elif mtype == "therapist_voice":
                b64 = data.get("data", "")
                if b64:
                    audio_bytes = base64.b64decode(b64)

                    tr = client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=("audio.webm", audio_bytes)
                    )
                    await ws.send_json({"type": "therapist_transcript", "text": tr.text})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("Voice socket error:", e)

    # cleanup
    finally:
        task.cancel()
        try:
            await openai_ws.close()
        except:
            pass

        state.end_time = time.time()
        record_billing(state, model_name)
# -------------------------------------------------------
# MAIN ENTRYPOINT
# -------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # production-ready host & port
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    
    # reload only in dev
    reload_flag = os.environ.get("DEV_RELOAD", "true").lower() in ["1", "true", "yes"]

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level="info",
    )
