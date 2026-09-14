import os
from typing import List, Dict, Any

class DocumentLoader:
    """
    Ingests local documents in TXT, PDF, and DOCX formats while preserving metadata.
    """
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir

    def load_txt(self, filepath: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return [{
            "text": content,
            "metadata": {
                "document_name": filename,
                "document_path": filepath,
                "page": 1,
                "format": "txt"
            }
        }]

    def load_pdf(self, filepath: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(filepath)
        documents = []
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "document_name": filename,
                            "document_path": filepath,
                            "page": page_idx + 1,
                            "format": "pdf"
                        }
                    })
        except Exception as e:
            print(f"Warning: Error reading PDF {filepath}: {e}")
        return documents

    def load_docx(self, filepath: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(filepath)
        try:
            import docx
            doc = docx.Document(filepath)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return [{
                "text": full_text,
                "metadata": {
                    "document_name": filename,
                    "document_path": filepath,
                    "page": 1,
                    "format": "docx"
                }
            }]
        except Exception as e:
            print(f"Warning: Error reading DOCX {filepath}: {e}")
            return []

    def load_directory(self) -> List[Dict[str, Any]]:
        documents = []
        if not os.path.exists(self.documents_dir):
            return documents

        for root, _, files in os.walk(self.documents_dir):
            for file in files:
                filepath = os.path.join(root, file)
                ext = file.lower().split('.')[-1]
                if ext in ['txt', 'md']:
                    documents.extend(self.load_txt(filepath))
                elif ext == 'pdf':
                    documents.extend(self.load_pdf(filepath))
                elif ext in ['docx', 'doc']:
                    documents.extend(self.load_docx(filepath))
        return documents
