# Therapy Chat Demo (Pilot)

## Overview

A supervised, privacy-focused GPT-powered therapy assistance tool built with FastAPI and OpenAI's APIs. The application provides real-time text and voice-based therapeutic conversations under licensed therapist supervision. The system prioritizes patient privacy by design—no Protected Health Information (PHI), transcripts, messages, or audio recordings are stored. Only minimal billing metadata (session_id, duration, model, timestamp) is retained.

## Recent Changes (December 2025)

- TTS Voice: Added calming prefix ("Please speak slowly, calmly, and gently in a therapeutic tone:") and using shimmer voice
- TTS Playback: Client-side playback slowed to 0.85x for calmer audio
- Realtime Voice: Fixed chipmunk effect by using proper 24kHz buffer sample rate with browser's default AudioContext, playbackRate 0.9

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture

**Framework**: FastAPI with WebSocket support for real-time bidirectional communication

**Core Components**:
- **WebSocket Endpoints**: Two primary endpoints (`/ws/chat` for text, `/ws/voice` for audio) handle real-time streaming communication
- **HTTP Endpoints**: REST endpoints for text-to-speech (`/api/tts`), speech-to-text (`/api/stt`), and health checks
- **Session Management**: In-memory session tracking using unique session IDs (UUID-based), no persistent storage of clinical content

**Privacy-First Design Philosophy**:
- Zero persistence of therapeutic content (conversations, audio, transcripts)
- Only anonymized billing/usage metadata stored
- No PHI (Protected Health Information) retention
- Safety gates implemented for input validation and output filtering

**AI Integration Pattern**:
- Single OpenAI client instance (lazy initialization pattern)
- Master therapy prompt defines AI behavior and boundaries
- Enforced English-only responses with gentle redirection for other languages
- Streaming token delivery for low-latency user experience

**Therapist Supervision Model**:
- Whisper command system for invisible therapist interventions (STOP, PAUSE/RESUME, TERMINATE, REDIRECT, CHANGE_TONE, LIMIT_DEPTH)
- Visible therapist messages displayed to patient
- Full transcript visibility for supervisor without storage

### Frontend Architecture

**Technology**: Single-page application with vanilla JavaScript (no framework dependencies)

**UI Components**:
- Text chat interface with streaming message display
- Voice session interface with continuous audio streaming
- Therapist control panel with whisper commands
- Real-time status indicators and latency metrics

**Communication Pattern**:
- WebSocket connections for real-time streaming (text tokens and audio chunks)
- Audio playback with controllable playback rate (0.85x for therapeutic pacing)
- Asynchronous API calls for TTS/STT conversion

**Design Principles**:
- Calm, professional aesthetic with minimal distractions
- Clear role differentiation (Patient, AI Assistant, Therapist)
- Accessibility-focused with status indicators and controls

### Data Flow

1. **Text Mode**: Patient input → Input safety gate → OpenAI GPT API (streaming) → Output safety gate → WebSocket delivery → Frontend display
2. **Voice Mode**: Patient audio → OpenAI Realtime API → Transcription + AI response (streaming) → Audio playback
3. **TTS Enhancement**: AI text response → Calm tone prefix injection → OpenAI TTS API ("shimmer" voice) → Slowed playback (0.85x)

### Safety Architecture

**Input Gate**: Filters crisis content, PHI, and inappropriate material before AI processing

**Output Gate**: Prevents AI from providing diagnoses, treatment plans, risk assessments, or operating outside therapeutic boundaries

**Supervision Layer**: Licensed therapist can intervene at any point with visible or invisible controls

## External Dependencies

### APIs and Services

**OpenAI API** (Primary dependency):
- **GPT Models**: Text-based therapeutic conversation (streaming completion)
- **Realtime API**: Voice session handling with continuous audio streaming
- **TTS (Text-to-Speech)**: Model "tts-1" with "shimmer" voice for calm, therapeutic audio output
- **Whisper (Speech-to-Text)**: Audio transcription for voice sessions
- Authentication via `OPENAI_API_KEY` environment variable

### Python Dependencies

- **fastapi**: Web framework for REST and WebSocket endpoints
- **uvicorn**: ASGI server for running FastAPI application
- **openai**: Official OpenAI Python client library
- **jinja2**: Template engine for serving HTML frontend
- **python-multipart**: File upload handling for audio endpoints
- **websockets**: WebSocket protocol implementation

### Environment Configuration

**Required Environment Variables**:
- `OPENAI_API_KEY`: OpenAI API authentication key

### Frontend Dependencies

**Browser APIs**:
- WebSocket API for real-time communication
- Web Audio API for audio playback control
- MediaRecorder API (potential use for voice capture)
- Fetch API for HTTP requests

**No external JavaScript libraries**: Vanilla JavaScript implementation for minimal dependencies and maximum control