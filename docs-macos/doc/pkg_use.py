from kitten_inference.tts08 import KittenTTS

tts = KittenTTS("KittenML/kitten-tts-mini-0.8")
audio = tts.generate("This high-quality voice runs without a GPU.", voice="Jasper")  # float32, 24 kHz
tts.generate_to_file("Hello, world.", "hello.wav", voice="Luna", speed=1.2)
