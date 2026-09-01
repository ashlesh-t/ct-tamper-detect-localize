import sys
from pathlib import Path

# Make the `pipeline` package importable, matching how the Streamlit app
# resolves it (prototype/ct-tampering-detector is the app's working dir).
PROTOTYPE_DIR = Path(__file__).resolve().parents[2] / "prototype" / "ct-tampering-detector"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
