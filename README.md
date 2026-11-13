# Voice Studio Web

Local web application for internal speech synthesis based on OmniVoice. The web server and model run on the same host; the browser provides the interface.

## Windows (Python 3.11, CPU)

Open PowerShell in the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/k2-fsa/OmniVoice.git
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. First model load downloads weights and requires network access and enough disk space. Run `python -m pytest -q` for startup/API tests; they do not download model weights.

## GPU

For NVIDIA, install a PyTorch CUDA build that matches the installed driver using the official PyTorch installation selector, instead of the CPU-wheel command above. The app can then select `CUDA` or `Auto`. On Apple Silicon, install a compatible PyTorch build and select `Apple Metal` or `Auto`.

## Data and safe use

Outputs and history are saved under the current user's application-data directory. Uploaded reference audio is temporary. The server defaults to localhost only; do not expose it on a network without authentication and a security review. Use only voice samples for which you have permission and comply with the OmniVoice model and code licenses.
