import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("data/chroma", exist_ok=True)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio

try:
    from app.core.config import settings
    from app.services.rag_engine import rag_engine
    RAG_AVAILABLE = True
    logger.info("RAG engine loaded successfully")
except Exception as e:
    logger.warning(f"RAG engine not available: {e}")
    RAG_AVAILABLE = False
    settings = None

app = FastAPI(title="智识游侠 RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(static_dir, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>智识游侠 - AI学习助手</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 80px; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
            button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin-top: 10px; }
            button:hover { background: #45a049; }
            #result { margin-top: 20px; white-space: pre-wrap; }
            .status { color: #666; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>🎮 智识游侠 - AI学习助手</h1>
        <div class="card">
            <p>基于 RAG + LangGraph 的智能问答系统</p>
            <textarea id="question" placeholder="请输入你的问题...">什么是机器学习？</textarea>
            <br>
            <button onclick="ask()">提交问题</button>
            <div class="status" id="status"></div>
            <div id="result"></div>
        </div>
        <script>
        async function ask() {
            const question = document.getElementById('question').value;
            const status = document.getElementById('status');
            const result = document.getElementById('result');
            status.textContent = '正在思考中...';
            result.textContent = '';
            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question, lecture_id: 'GLOBAL_SEARCH'})
                });
                const data = await res.json();
                status.textContent = '';
                result.textContent = data.answer || data.error || '无响应';
            } catch(e) {
                status.textContent = '';
                result.textContent = '错误: ' + e.message;
            }
        }
        </script>
    </body>
    </html>
    """

@app.get("/exam")
async def exam_page():
    html_path = os.path.join(static_dir, "exam.html")
    if os.path.exists(html_path, "r", encoding="utf-8") as f:
        return f.read()
    return "<h1>考试页面</h1><p>exam.html 不存在</p>"

@app.get("/health")
def health():
    return {"status": "ok", "rag_available": RAG_AVAILABLE}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from 智识游侠 backend!", "rag": RAG_AVAILABLE}

class AskRequest(BaseModel):
    question: str
    lecture_id: str = "GLOBAL_SEARCH"
    image_base64: Optional[str] = None

@app.post("/api/ask")
async def ask(request: AskRequest):
    if not RAG_AVAILABLE:
        return {"answer": "RAG 服务暂不可用，请检查环境配置", "error": "RAG not available"}
    
    try:
        logger.info(f"Received question: {request.question[:50]}...")
        answer = await rag_engine.get_answer(
            question=request.question,
            lecture_id=request.lecture_id,
            image_base64=request.image_base64
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in ask: {e}")
        return {"answer": f"处理出错: {str(e)}", "error": str(e)}

@app.get("/api/ask/stream")
async def ask_stream(q: str, lecture_id: str = "GLOBAL_SEARCH"):
    if not RAG_AVAILABLE:
        return {"answer": "RAG 服务暂不可用"}
    
    async def event_generator():
        try:
            async for chunk in rag_engine.get_answer_stream(question=q, lecture_id=lecture_id):
                yield chunk
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/check/{lecture_id}")
def check_lecture(lecture_id: str):
    if not RAG_AVAILABLE:
        return {"ingested": False, "chunks": []}
    
    try:
        ingested = rag_engine.check_ingested(lecture_id)
        chunks = rag_engine.get_namespace_chunks(lecture_id) if ingested else []
        return {"ingested": ingested, "chunks": len(chunks)}
    except Exception as e:
        return {"ingested": False, "error": str(e)}

import json

handler = app