from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="智识游侠")

static_dir = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head><title>智识游侠</title></head>
    <body>
        <h1>智识游侠 - AI学习助手</h1>
        <button onclick="testApi()">点击测试后端</button>
        <p id="result"></p>
        <script>
        async function testApi() {
            try {
                const res = await fetch('/api/hello');
                const data = await res.json();
                document.getElementById('result').innerText = '后端返回: ' + data.message;
            } catch(e) {
                document.getElementById('result').innerText = '错误: ' + e.message;
            }
        }
        </script>
    </body>
    </html>
    """

@app.get("/exam")
async def exam_page():
    html_path = os.path.join(static_dir, "exam.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>考试页面</h1><p>exam.html 不存在</p>"

@app.get("/api/hello")
def hello():
    return {"message": "Hello from 智识游侠 backend!"}

@app.get("/health")
def health():
    return {"status": "ok"}

handler = app