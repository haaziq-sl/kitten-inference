"""The Python packages Kitten TTS 0.8 needs (kitten-inference). They are looked up before the
first import of any of them, so a missing one gives one message that names them all and the install command, instead
of a bare "No module named ...". Text normalisation (normalize_text) needs none of them."""
import importlib.util

# Import names, in the order of the extra (which may spell espeakng_loader and huggingface_hub with "-").
REQUIRED = ("espeakng_loader", "phonemizer", "soundfile", "numpy", "huggingface_hub")


def missing(names=REQUIRED):
    """The modules in ``names`` that this Python cannot find (none of them is imported)."""
    lacking = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except ValueError:  # already imported, without a spec (put into sys.modules by hand)
            found = True
        if not found:
            lacking.append(name)
    return lacking


def require():
    """ImportError naming the missing packages and how to install them; nothing when all are there."""
    lacking = missing()
    if lacking:
        raise ImportError(
            f"Kitten TTS 0.8 needs more Python packages (not installed: {', '.join(lacking)}). Install them: "
            'pip install "kitten-inference".'
        )
