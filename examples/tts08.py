from kitten_inference import KittenTTS

# Models: kitten-tts-mini-0.8 (80M, best quality), kitten-tts-micro-0.8 (40M),
# kitten-tts-nano-0.8 (15M, fastest). Add "-int8" for the smaller int8 weights.
tts = KittenTTS("KittenML/kitten-tts-mini-0.8")
print(tts.available_voices)  # ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']

text = "One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp."
tts.generate_to_file(text, "output.wav", voice="Bruno")

# speed > 1.0 speaks faster, < 1.0 slower. generate() returns float32 samples at 24 kHz.
audio = tts.generate(text, voice="Luna", speed=1.2)
