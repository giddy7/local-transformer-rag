import re

class DocumentCleaner:
    """
    Cleans and normalizes text extracted from documents.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Normalize carriage returns and line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove non-printable control characters except newline and tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Collapse multiple spaces into single space per line
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        # Collapse multiple blank lines into max 2 newlines
        cleaned = '\n'.join(lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
