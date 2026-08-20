# Dil — Multi-Modal Cardiovascular Risk Diagnostic Platform

A dual-delivery cardiovascular risk platform with a Streamlit clinical sandbox and FastAPI developer API.

## Project Structure

```
├── app/                 # Main application entry point
├── api/                 # FastAPI routes and middleware
├── db/                  # Database models and session management
├── models/              # ML model definitions and inference
├── streamlit_app/       # Streamlit clinical sandbox UI
├── tests/               # Unit and integration tests
├── docker/              # Docker configuration
└── requirements.txt     # Python dependencies
```

## Quick Start

1. Create virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment file:
   ```bash
   copy .env.example .env
   ```

4. Run the API server:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Run the Streamlit sandbox (in another terminal):
   ```bash
   streamlit run streamlit_app/main.py
   ```

## API Endpoints

- `GET /healthz` - Health check
- `POST /api/v1/predict-tabular` - Tabular risk prediction (requires X-API-Key)
- `POST /api/v1/predict-image` - Imaging risk prediction (requires X-API-Key)

## Documentation

See `Dil_PRD.md` and `Dil_TDR.md` for full product and technical specifications.