"""Set JAVA_HOME for pyjnius before Pyserini is imported."""

import os
import shutil
from pathlib import Path


def ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    java = shutil.which("java")
    if java:
        os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)
