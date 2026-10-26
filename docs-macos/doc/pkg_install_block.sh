python3 -m venv .venv && . .venv/bin/activate   # Windows, in a Command Prompt: py -m venv .venv, then .venv\Scripts\activate.bat
python -m pip install -U pip                    # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"
