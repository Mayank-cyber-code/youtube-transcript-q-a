from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from qa_engine import YouTubeConversationalQA

app = FastAPI()
qa = YouTubeConversationalQA()

# Allow CORS for web/extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use strict origins for production!
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/frontend.html")

@app.post('/api/ask')
async def ask(request: Request):
    data = await request.json()
    video_url = data['video_url']
    question = data['question']
    answer = qa.ask(video_url, question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        reload=True
    )
