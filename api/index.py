from fastapi import FastAPI
app = FastAPI()

@app.get("/api/hello")
def read_root():
    return {"message": "Hello！这是来自 Vercel 云端 Python 后端的问候！"}