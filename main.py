import os
import logging
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.environ.get("API", "")
API_BASE = os.environ.get("API_BASE", "https://api.openai.com/v1")

app = FastAPI(title="智识游侠")

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
        <h1>智识游侠 - AI学习助手</h1>
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

@app.get("/health")
def health():
    return {"status": "ok", "service": "智识游侠"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from 智识游侠 backend!"}

@app.post("/api/ask")
async def ask(request: dict):
    question = request.get("question", "")
    lecture_id = request.get("lecture_id", "GLOBAL_SEARCH")
    
    if not API_KEY:
        return {"answer": "错误: 未配置 API Key", "sources": []}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "你是一个智能学习助手，专门帮助用户解答问题。"},
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 500
                }
            )
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "无响应")
            return {"answer": answer, "sources": []}
    except Exception as e:
        return {"answer": f"请求失败: {str(e)}", "sources": []}

handler = app