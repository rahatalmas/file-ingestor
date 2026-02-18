"""
File Watcher - Monitors nested folder structure for new files
Expected path: <base>/<company_id>/notprocessed/<filename>
"""

import time
import os
import re
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from docling_processor.ingestor import ingest_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("file_watcher")

# Regex to extract company_id from path
# Expected: .../butter-app-knowledge-files/<company_id>/notprocessed/<file>
COMPANY_ID_PATTERN = re.compile(
    r"[/\\]([a-f0-9\-]{36})[/\\]notprocessed[/\\]",
    re.IGNORECASE
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".txt", ".md", ".html",
    ".htm", ".csv", ".json", ".xml", ".rtf",
    ".odt", ".ods", ".odp", ".epub"
}


def extract_company_id(file_path: str) -> str | None:
    """Extract company UUID from file path"""
    match = COMPANY_ID_PATTERN.search(file_path)
    if match:
        return match.group(1)
    # Fallback: try to find UUID-like segment in path
    parts = re.split(r"[/\\]", file_path)
    uuid_re = re.compile(
        r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        re.IGNORECASE
    )
    for part in parts:
        if uuid_re.match(part):
            return part
    return None


class FileUploadHandler(FileSystemEventHandler):
    """Handles file system events and triggers ingestion"""

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_file(event.src_path)

    def on_moved(self, event):
        """Handle files moved into the watched folder"""
        if event.is_directory:
            return
        self._handle_file(event.dest_path)

    def _handle_file(self, path: str):
        full_path = os.path.abspath(path)
        ext = os.path.splitext(full_path)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.info(f"Skipping unsupported file type: {full_path}")
            return

        # Only process files inside a 'notprocessed' directory
        if "notprocessed" not in full_path.replace("\\", "/"):
            logger.debug(f"Ignoring file not in notprocessed dir: {full_path}")
            return

        company_id = extract_company_id(full_path)
        if not company_id:
            logger.warning(f"Could not extract company_id from path: {full_path}")
            return

        logger.info(f"New file detected | company={company_id} | file={full_path}")

        # Give OS time to finish writing the file
        time.sleep(1.5)

        ingest_file(full_path, company_id)


def watch_folder(folder_path: str):

    """
    Watch a folder recursively for new file uploads.

    Args:
        folder_path: Root path to monitor
    """
    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        logger.error(f"Folder does not exist: {folder_path}")
        return

    if not os.path.isdir(folder_path):
        logger.error(f"Path is not a directory: {folder_path}")
        return

    logger.info(f"Starting file watcher: {folder_path}")
    logger.info("Press Ctrl+C to stop...\n")

    event_handler = FileUploadHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()

    observer.join()
    logger.info("File watcher stopped.")