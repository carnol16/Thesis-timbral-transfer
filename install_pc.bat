@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo   Starting PC Timbre Transfer Setup...
echo ==========================================

:: ---- Find a compatible Python version ----
set PYTHON_CMD=
for %%V in (3.12 3.11 3.10) do (
    py -%%V --version >nul 2>&1
    if !errorlevel! == 0 (
        set PYTHON_CMD=py -%%V
        echo Using Python: py -%%V
        goto :found_python
    )
)

:: Fallback: try plain "python" and check version
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    echo Using Python: python
    goto :found_python
)

echo ERROR: Could not find Python 3.10, 3.11, or 3.12.
echo PyTorch does not support Python 3.13+ yet.
echo Please install Python 3.12 from https://www.python.org/downloads/
pause
exit /b 1

:found_python

echo.
echo 1/4: Setting up the main Virtual Environment (.venv)...
%PYTHON_CMD% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements_pc.txt
call .venv\Scripts\deactivate.bat

echo.
echo 2/4: Setting up the Stable Audio Venv (sa_venv)...
%PYTHON_CMD% -m venv sa_venv
call sa_venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA 12.4 support...
pip uninstall -y torchvision 2>nul
python -c "import torch; exit(0 if '+cu' in torch.__version__ else 1)" 2>nul
if !errorlevel! neq 0 (
    echo Ensuring CUDA-enabled PyTorch in sa_venv...
    pip uninstall -y torch torchaudio 2>nul
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
) else (
    echo PyTorch CUDA build already verified in sa_venv.
)
pip install numpy scipy soundfile librosa torchcodec einops einops-exts safetensors transformers huggingface-hub tqdm
call sa_venv\Scripts\deactivate.bat

echo.
echo 3/4: Setting up stable-audio-3 for the Autoencoder...
if exist "stable-audio-3" goto :sa_ready

echo Downloading stable-audio-3 repository into root folder...
git clone https://github.com/Stability-AI/stable-audio-3.git
if exist "stable-audio-3" goto :sa_ready

echo Git clone failed or Git is not installed. Trying fallback download...
call sa_venv\Scripts\activate.bat
python -c "import urllib.request, zipfile, os; urllib.request.urlretrieve('https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip', 'sa3.zip'); zipfile.ZipFile('sa3.zip').extractall('.'); os.rename('stable-audio-3-main', 'stable-audio-3'); os.remove('sa3.zip')"
call sa_venv\Scripts\deactivate.bat

:sa_ready
if not exist "stable-audio-3" (
    echo ERROR: Could not find or download stable-audio-3.
    echo Please make sure Git is installed or manually clone:
    echo   git clone https://github.com/Stability-AI/stable-audio-3.git
    echo into this folder, then re-run install_pc.bat.
    pause
    exit /b 1
)

call sa_venv\Scripts\activate.bat
echo Installing stable-audio-3 (using existing PyTorch CUDA build)...
pip install --no-deps -e stable-audio-3
call sa_venv\Scripts\deactivate.bat

echo.
echo 4/4: Checking and downloading model checkpoint (diT_best_val.pt)...
call .venv\Scripts\activate.bat
python download_checkpoint.py
call .venv\Scripts\deactivate.bat

echo.
echo ==========================================
echo   Setup Complete! You are ready to run!
echo.
echo   To start the GUI, open a new terminal and run:
echo      .venv\Scripts\activate
echo      python run_app.py
echo ==========================================
pause

