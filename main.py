"""
Enterprise RAG Pipeline - Main Entry Point
Watches a folder for new files and ingests them into Qdrant (multi-tenant)
"""

import os
from file_watcher.watcher import watch_folder
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    WATCH_FOLDER = "../butter-app-knowledge-files"
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    watch_folder(WATCH_FOLDER)