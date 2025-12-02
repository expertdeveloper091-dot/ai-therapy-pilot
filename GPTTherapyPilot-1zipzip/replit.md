# Therapy Chat Demo (Pilot)

## Overview
A supervised, privacy-focused, low-latency GPT therapy tool demo. This application demonstrates:

- **AI Assistant ("chat")**: Speaks with a patient in a therapy-like style using OpenAI's GPT model
- **Licensed Therapist Supervision**: Can see full transcript, send visible messages, and send hidden "whisper" commands
- **Real-time Communication**: Streaming text responses and voice sessions
- **Privacy-First Design**: No transcripts, messages, or audio are stored - only billing metadata

## Project Structure
```
/
├── main.py              # FastAPI backend with WebSocket endpoints
├── templates/
│   └── index.html       # Calm, professional single-page UI
├── requirements.txt     # Python dependencies
└── replit.md           # This file
```

## Features

### Text Chat Mode
- Streaming token-by-token responses
- Latency metrics (TTFT + full response time)
- Text-to-speech playback of AI responses

### Voice Session Mode
- Continuous, hands-free voice conversation
- Real-time audio streaming via OpenAI Realtime API
- Automatic transcription display

### Safety Gates
- **Input Gate**: Blocks crisis content and PHI (Protected Health Information)
- **Output Gate**: Prevents AI from providing diagnoses, treatment plans, etc.

### Therapist Controls (Whisper Commands)
- STOP: Stop current AI response
- PAUSE/RESUME: Pause/resume session
- TERMINATE: End session completely
- REDIRECT: Inject hidden instruction
- CHANGE_TONE: Adjust AI's communication style
- LIMIT_DEPTH: Keep responses surface-level

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/health` | GET | Health check |
| `/billing-debug` | GET | View billing records (demo) |
| `/api/tts` | POST | Text-to-speech conversion |
| `/api/stt` | POST | Speech-to-text conversion |
| `/ws/chat` | WebSocket | Streaming text chat |
| `/ws/voice` | WebSocket | Real-time voice session |

## Environment Variables
- `OPENAI_API_KEY`: Required. Your OpenAI API key for GPT and audio services.

## Privacy Principles
- No PHI (Protected Health Information) stored
- No transcripts, messages, or audio stored
- Only billing metadata retained (session_id, duration, model, timestamp)
- No clinical content logged

## Running the App
The app runs on port 5000. Start with:
```bash
python main.py
```

## Recent Changes
- Initial implementation with full feature set
- Calm, professional UI with role-based message styling
- WebSocket support for text and voice modes
- Safety gates for input/output filtering
- Therapist control panel with whisper commands
