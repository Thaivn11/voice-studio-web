# Voice Studio Web

Local-only web application for internal speech generation with OmniVoice as the inference engine. The browser is the interface; audio inference runs on the computer hosting the server.

## Windows (PowerShell, Python 3.11 64-bit)

Open PowerShell in the cloned repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch for the target device. CPU installation example:

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPU, use the official PyTorch installation selector to choose the CUDA wheel compatible with the installed driver instead of the CPU command. Apple Silicon users should install the compatible PyTorch build for MPS.

Install OmniVoice and start the server:

```powershell
python -m pip install git+https://github.com/k2-fsa/OmniVoice.git
$env:VOICE_STUDIO_ADMIN_PASSWORD = "Choose-A-Long-Unique-Password"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. If this is the first run and the password environment variable was not set, the server prints a random one-time administrator password in its console. Save it before closing that terminal. Use username `admin`. The model is loaded only after signing in and pressing **Load model**; the first load may download a large model and requires internet access.

## Checks

```powershell
python -m pytest -q
python -m pip check
```

These smoke tests do not load model weights or prove that real inference works on a specific machine. For an inference test, load the model in the UI and synthesize a short text using an audio sample you are authorized to use.

## Data and deployment warning

Generated audio, project metadata, user hashes, and history are stored locally under the user's application-data directory. Uploaded audio is temporary for generation; project sample audio is retained locally. Sessions expire after 12 hours and require signing in again after server restart. This is a localhost development build: do not expose it on a LAN or the public internet without a security review, TLS, stronger operational controls, and a proper database. Use only voices for which you have permission and comply with the OmniVoice model/code licenses.
