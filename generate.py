"""兼容入口，转发到 `src/gan_faces/infer/generate.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.infer.generate import main


if __name__ == "__main__":
    main()
