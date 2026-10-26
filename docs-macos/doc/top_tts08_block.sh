python3 -m venv .venv && . .venv/bin/activate   # Windows, in a Command Prompt: py -m venv .venv, then .venv\Scripts\activate.bat
python -m pip install -U pip                    # pip older than 23.3 skips "[tts-0.8]" with only a warning
pip install "kitten-inference[tts-0.8]"
python -c "from kitten_inference.tts08 import KittenTTS; KittenTTS('KittenML/kitten-tts-nano-0.8').generate_to_file('Hello from Kitten TTS.', 'hello.wav', voice='Jasper')"
