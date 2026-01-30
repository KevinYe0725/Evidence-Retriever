# Evidence Retrieval (MVP)

一个最小可用的“证据检索器”服务：导入文本后自动切分为可引用的 chunks，调用外部 Qwen Embedding API 生成向量写入本地 ChromaDB，并提供语义检索返回证据片段与引用定位信息。

## 功能概览
- 文本导入 -> 分段切分 + 向量化 -> 写入 MySQL/Chroma
- 语义检索（相似度）返回证据列表（citation + excerpt + location）
- 软删除与恢复（通过 is_deleted 过滤）

## 依赖
- Python 3.11
- MySQL 8+

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置
复制并编辑 `.env`：
```bash
cp .env.example .env
```

说明：
- `QWEN_EMBEDDING_*` 需按官方文档补齐真实调用。
- 如需本地快速跑通流程，可将 `QWEN_EMBEDDING_BASE_URL=mock`（会生成可复现的假向量用于调试）。
- 启用重排序：设置 `ENABLE_RERANK=true`，并配置 `QWEN_RERANK_BASE_URL`、`QWEN_RERANK_MODEL`。
- 当 rerank 调用失败时，会自动降级回向量检索顺序，不会导致搜索接口报错。

## 建表 SQL
```sql
CREATE TABLE docs (
  doc_id VARCHAR(64) PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  tags_json JSON NULL,
  raw_text LONGTEXT NOT NULL,
  is_deleted TINYINT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
  chunk_id VARCHAR(64) PRIMARY KEY,
  doc_id VARCHAR(64) NOT NULL,
  para_index INT NOT NULL,
  char_start INT NOT NULL,
  char_end INT NOT NULL,
  text TEXT NOT NULL,
  is_deleted TINYINT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_chunks_doc_id (doc_id)
);

CREATE TABLE embeddings_meta (
  id INT AUTO_INCREMENT PRIMARY KEY,
  doc_id VARCHAR(64) NOT NULL,
  embedding_provider VARCHAR(64) NOT NULL,
  embedding_model VARCHAR(255) NOT NULL,
  embedding_dim INT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_embeddings_meta_doc_id (doc_id)
);
```

## 启动
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Curl 示例
### 1) 导入文档
```bash
curl -X POST http://localhost:8000/v1/docs \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "用户输入-001",
    "text": "第一段内容。\n\n第二段内容，包含更多细节。",
    "tags": ["demo", "test"]
  }'
```

### 2) 语义检索
```bash
curl -X POST http://localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "第二段讲了什么？",
    "top_k": 20,
    "return_n": 6,
    "filters": {"doc_ids": [], "tags_any": []},
    "include_context": false
  }'
```

### 3) 删除与恢复
```bash
curl -X DELETE http://localhost:8000/v1/docs/{doc_id}
curl -X POST http://localhost:8000/v1/docs/{doc_id}/restore
```

## API 列表
- `POST /v1/docs`
- `GET /v1/docs/{doc_id}`
- `GET /v1/docs`
- `DELETE /v1/docs/{doc_id}`
- `POST /v1/docs/{doc_id}/restore`
- `POST /v1/search`
- `GET /v1/chunks/{chunk_id}`
- `GET /v1/chunks/{chunk_id}/context?before=1&after=1`
- `GET /health`

## 备注
- 软删除通过 `is_deleted` 控制，Chroma 查询只过滤 `is_deleted=0`。
- `core/reranker.py` 按 Qwen Rerank API 文档实现，如需调整字段请以官方文档为准。
