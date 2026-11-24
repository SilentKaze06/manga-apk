import os
import sys

# Ensure project root, server/ml, and server are always in sys.path when running pytest
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
SERVER_DIR = os.path.join(REPO_ROOT, "manga_apk", "server")
SERVER_ML_DIR = os.path.join(SERVER_DIR, "ml")

for p in [PROJECT_ROOT, SERVER_ML_DIR, SERVER_DIR, REPO_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
