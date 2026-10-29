from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from build_support import hooks
build_wheel, build_sdist = hooks('tts2')
