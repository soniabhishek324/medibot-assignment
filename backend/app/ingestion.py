from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.rbac import COLLECTION_ROLES


SUPPORTED_SUFFIXES = {".pdf", ".md"}


@dataclass
class DocumentChunk:
    id: str
    text: str
    source_document: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str = "text"

    @property
    def metadata(self) -> dict:
        return {
            "source_document": self.source_document,
            "collection": self.collection,
            "access_roles": self.access_roles,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
        }


def discover_documents(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)


def parse_document(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def infer_collection(path: Path, data_dir: Path) -> str:
    relative = path.relative_to(data_dir)
    return relative.parts[0]


def normalize_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return stripped.lstrip("#").strip()
    if len(stripped) <= 90 and (stripped.isupper() or re.match(r"^(\d+\.?\s+)?[A-Z][A-Za-z0-9 /&(),:-]+$", stripped)):
        return stripped
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = normalize_heading(line)
        if heading and len(sections[-1][1]) > 2:
            sections.append((heading, []))
        else:
            sections[-1][1].append(raw_line)
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def chunk_text(body: str, max_words: int = 220, overlap: int = 35) -> list[str]:
    words = body.split()
    if len(words) <= max_words:
        return [body.strip()]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def detect_chunk_type(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    tableish = sum(1 for line in lines if "|" in line or re.search(r"\s{2,}", line))
    if tableish >= 3:
        return "table"
    if text.strip().startswith("```"):
        return "code"
    return "text"


def build_chunks(data_dir: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in discover_documents(data_dir):
        collection = infer_collection(path, data_dir)
        if collection not in COLLECTION_ROLES:
            continue
        text = parse_document(path)
        for section_title, section_body in split_sections(text):
            for ordinal, piece in enumerate(chunk_text(section_body)):
                embedded_text = f"Section: {section_title}\nSource: {path.name}\n\n{piece}"
                digest = hashlib.sha1(f"{path}:{section_title}:{ordinal}:{piece[:80]}".encode()).hexdigest()
                chunks.append(
                    DocumentChunk(
                        id=digest,
                        text=embedded_text,
                        source_document=path.name,
                        collection=collection,
                        access_roles=COLLECTION_ROLES[collection],
                        section_title=section_title,
                        chunk_type=detect_chunk_type(piece),
                    )
                )
    return chunks
