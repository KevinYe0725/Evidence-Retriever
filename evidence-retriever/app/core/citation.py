from app.db.models import Chunk


def build_citation(doc_title: str, para_index: int) -> str:
    return f"[{doc_title}#P{para_index}]"


def build_excerpt(text: str, max_len: int = 200) -> str:
    snippet = text.strip()
    if len(snippet) <= max_len:
        return snippet
    return snippet[:max_len].rstrip() + "..."


def build_location(chunk: Chunk) -> dict:
    return {
        "para_index": chunk.para_index,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


def build_citation_bundle(doc_title: str, chunk: Chunk) -> tuple[str, str, dict]:
    citation = build_citation(doc_title, chunk.para_index)
    excerpt = build_excerpt(chunk.text)
    location = build_location(chunk)
    return citation, excerpt, location
