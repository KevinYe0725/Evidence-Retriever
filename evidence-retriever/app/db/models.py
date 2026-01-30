from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.dialects.mysql import LONGTEXT, TINYINT


Base = declarative_base()


class Doc(Base):
    __tablename__ = "docs"

    doc_id = Column(String(64), primary_key=True)
    title = Column(String(255), nullable=False)
    tags_json = Column(MySQLJSON, nullable=True)
    raw_text = Column(LONGTEXT, nullable=False)
    is_deleted = Column(TINYINT, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String(64), primary_key=True)
    doc_id = Column(String(64), index=True, nullable=False)
    para_index = Column(Integer, nullable=False)
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    is_deleted = Column(TINYINT, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EmbeddingMeta(Base):
    __tablename__ = "embeddings_meta"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), index=True, nullable=False)
    embedding_provider = Column(String(64), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
