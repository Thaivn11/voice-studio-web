# Voice Studio Web

A private internal web application for voice cloning and speech synthesis built around OmniVoice-style inference workflows.

## Project structure

- `app/` – Python backend package
- `app/api.py` – FastAPI app entry point
- `app/config.py` – runtime configuration and local storage
- `app/core/` – device detection and inference logic
- `app/static/` – frontend assets
- `tests/` – smoke tests

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.api
```

Open http://127.0.0.1:8000

## Notes

- Default host is localhost only.
- Model loads on CPU or GPU depending on hardware and selected device.
- Use only authorized voice samples.
