from docling.document_converter import DocumentConverter
from docling_processor import chunk_creator,chunk_creator_pro
import json

def start_ingesting(file: str):
    converter = DocumentConverter()
    converted_doc = converter.convert(file)
    doc = converted_doc.document
    doc_json = doc.export_to_dict()
    # print(doc_json)
    with open("output.json", "w") as f:
        json.dump(doc_json, f, indent=2)
    
    # chunks = chunk_creator.chunk_docling_json(doc_json,"test-1")
    chunks = chunk_creator.chunk_docling_json(doc_json,"test-arong-1")
    print(chunks)
    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)