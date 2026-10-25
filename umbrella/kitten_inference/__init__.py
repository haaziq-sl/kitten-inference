"""Kitten TTS inference engines.

This package holds no engine. Each engine is its own distribution, installed with an extra:
``pip install "kitten-inference[tts-0.8]"`` (``kitten_inference.tts08``), ``"kitten-inference[tts-2]"``
or ``"kitten-inference[tts-2-rocm]"`` (``kitten_inference.tts2``).
"""

import pkgutil

__version__ = "0.2.0"

# The engines install into kitten_inference/ from their own distributions; extend_path also finds them when they
# are installed in another sys.path entry than this file.
__path__ = pkgutil.extend_path(__path__, __name__)

del pkgutil
