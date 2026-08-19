"""Medical report ingestion.

Turns an uploaded report (PDF or plain text) into the raw text that
`backend.summarize_report` works from. No interpretation happens here -- this
module only extracts characters.
"""

SUPPORTED_TYPES = ["pdf", "txt", "md", "text"]

# Reports longer than this are trimmed before being sent to the model.
CHAR_LIMIT = 60000


def _read_pdf(data):
    try:
        import pymupdf
    except ImportError:  # older installs only expose the deprecated alias
        import fitz as pymupdf

    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def _read_text(data):
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text(filename, data):
    """Extract text from an uploaded report.

    Returns (text, note). `note` is a short human-readable line about what was
    read, or an error message when extraction failed.
    """
    if not data:
        return "", "The uploaded file is empty."

    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            text = _read_pdf(data)
            kind = "PDF"
        else:
            text = _read_text(data)
            kind = "text file"
    except Exception as exc:
        return "", f"Could not read the file: {exc}"

    text = text.strip()
    if not text:
        return "", ("No selectable text found in this PDF. It is most likely a "
                    "scan -- OCR would be needed, which is not connected here.")

    words = len(text.split())
    return text, f"Read {words:,} words from the {kind}."


def preview(text, limit=600):
    """Short preview of the extracted text for the UI."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."
