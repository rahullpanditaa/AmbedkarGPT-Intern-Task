from pathlib import Path

DATA_DIR_PATH = Path(__file__).parent.parent.resolve() / "data"
SPEECH_TXT_PATH = DATA_DIR_PATH / "speech.txt"

CHROMA_DIR_PATH = Path(__file__).parent.resolve() / "chroma_vector_db"