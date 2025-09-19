# 🔧 Backend for Radiance

> *LangGraph-powered agentic system with multimodal AI capabilities*

[![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)

**Microservices API backend** providing **computer vision**, **voice processing**, and **agentic AI** capabilities for Radiance mobile app. <br>
🔗 **[Radiance Repository: ieee_sep2025](https://github.com/adonaydem/ieee-sep2025/)**  

## 🏗️ Architecture

```
sep_backend/
├── app.py                 # Flask API server & routing layer
├── chat_utils.py          # LangGraph agent + tool orchestration  
├── voice_chat.py          # Audio I/O & speech processing
├── ocr.py                 # Tesseract OCR pipeline
├── scenedescription.py    # ResNet-18 scene classification
├── directions.py          # Mapbox geolocation services
├── postgres_utils.py      # Database connection pooling
└── preferences.py         # User settings management
```

## 🚀 Quick Setup

```bash
# Install dependencies
pip install -r req.txt

# Configure environment
cp .env_example .env
# Add: OPENAI_API_KEY, ELEVENLABS_API_KEY, POSTGRES_SUPABASE, TAVILY_API_KEY, MAPBOX_ACCESS_TOKEN

# Run server
python app.py
```

## 🎯 API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/chat_audio` | POST | Voice-to-agent interaction |
| `/ocr` | POST | Tesseract text extraction |
| `/sd` | POST | Scene description analysis |
| `/send_voice_chat` | POST | P2P voice messaging |
| `/distress` | POST | Emergency alert system |
| `/api/preferences` | GET/POST | User configuration |

## 🧠 Core Components

**Agentic System (`chat_utils.py`)**
- **LangGraph ReAct agent** with tool-calling capabilities
- **PostgreSQL memory** for conversation persistence  
- **Function routing** for OCR, scene analysis, directions

**Computer Vision Pipeline**
- **Tesseract OCR** with preprocessing optimizations
- **ResNet-18 Places365** for scene classification
- **PIL + OpenCV** image processing stack

**Voice Processing**
- **SpeechRecognition** for STT
- **ElevenLabs** for natural TTS synthesis
- **Audio chunking** for real-time streaming

## 🛡️ Production Considerations

- **Connection pooling** via `psycopg_pool`
- **Resilient PostgreSQL** with automatic reconnection
- **Async processing** for non-blocking operations
- **File upload security** with `secure_filename()`

## 📊 Dependencies

**Core Framework**: Flask, LangChain 
**AI/ML**: OpenAI GPT-4o-mini, BLIP, Transformers, PyTorch  
**Database**: PostgreSQL with psycopg[binary,pool]  
**Audio**: ElevenLabs, SpeechRecognition, pydub  
**Vision**: OpenCV, Pillow, pytesseract  
**Maps**: Mapbox, geopy for geolocation services

