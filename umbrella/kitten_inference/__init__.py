"""Kitten TTS inference engines.

``kitten_inference.tts08``: Kitten TTS 0.8 on the CPU (``pip install "kitten-inference[tts-0.8]"``).
``kitten_inference.tts2``: Kitten TTS 2 on NVIDIA and AMD GPUs (``"kitten-inference[tts-2]"``); the GPU is chosen when
it starts. The extras add the Python packages each engine needs; the engines are in this package.
"""

import pkgutil

__version__ = "0.2.0"

# In a release wheel the engines sit beside this file. extend_path also finds them in another sys.path entry, as in
# source-tree runs (PYTHONPATH=umbrella:tts-0.8:tts-2/python) and with development engine builds.
__path__ = pkgutil.extend_path(__path__, __name__)

del pkgutil
