import re
from typing import List, Tuple

from app.schemas.chunk import ChunkCreate


_MAX_CHUNK_LEN = 1200
_PUNCTUATION = {"。", "；", ";", ".", "!", "！", "?", "？"}


def _has_blank_lines(text: str) -> bool:
    return bool(re.search(r"\n\s*\n", text))


def _iter_paragraph_spans(raw_text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    if not raw_text:
        return spans

    if _has_blank_lines(raw_text):
        pos = 0
        current_start = None
        for line in raw_text.splitlines(keepends=True):
            line_stripped = line.strip("\r\n")
            if line_stripped == "":
                if current_start is not None:
                    end = pos
                    text_end = end
                    while text_end > current_start and raw_text[text_end - 1] in "\r\n":
                        text_end -= 1
                    if text_end > current_start:
                        spans.append(
                            (current_start, text_end, raw_text[current_start:text_end])
                        )
                    current_start = None
            else:
                if current_start is None:
                    current_start = pos
            pos += len(line)
        if current_start is not None:
            end = len(raw_text)
            text_end = end
            while text_end > current_start and raw_text[text_end - 1] in "\r\n":
                text_end -= 1
            if text_end > current_start:
                spans.append((current_start, text_end, raw_text[current_start:text_end]))
    else:
        pos = 0
        for line in raw_text.splitlines(keepends=True):
            line_stripped = line.strip("\r\n")
            if line_stripped == "":
                pos += len(line)
                continue
            start = pos
            end = pos + len(line)
            text_end = end
            while text_end > start and raw_text[text_end - 1] in "\r\n":
                text_end -= 1
            if text_end > start:
                spans.append((start, text_end, raw_text[start:text_end]))
            pos += len(line)

    return spans


def _split_long_paragraph(text: str, base_start: int) -> List[Tuple[int, int, str]]:
    if len(text) <= _MAX_CHUNK_LEN:
        return [(base_start, base_start + len(text), text)]

    segments: List[Tuple[int, int, str]] = []
    seg_start = 0
    last_punct = -1
    for i, ch in enumerate(text):
        if ch in _PUNCTUATION:
            last_punct = i
        if i - seg_start + 1 >= _MAX_CHUNK_LEN:
            if last_punct >= seg_start:
                cut = last_punct + 1
            else:
                cut = i + 1
            segment_text = text[seg_start:cut]
            segments.append((base_start + seg_start, base_start + cut, segment_text))
            seg_start = cut
            last_punct = -1
    if seg_start < len(text):
        segment_text = text[seg_start:]
        segments.append((base_start + seg_start, base_start + len(text), segment_text))

    return segments


def chunk_text(raw_text: str) -> List[ChunkCreate]:
    chunks: List[ChunkCreate] = []
    para_index = 1
    for start, end, text in _iter_paragraph_spans(raw_text):
        if not text.strip():
            continue
        split_segments = _split_long_paragraph(text, start)
        for seg_start, seg_end, seg_text in split_segments:
            if not seg_text.strip():
                continue
            chunks.append(
                ChunkCreate(
                    para_index=para_index,
                    char_start=seg_start,
                    char_end=seg_end,
                    text=seg_text,
                )
            )
        para_index += 1
    return chunks
