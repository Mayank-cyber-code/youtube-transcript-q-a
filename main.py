from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from qa_engine import YouTubeConversationalQA

app = FastAPI()
qa = YouTubeConversationalQA()

# Allow CORS for web/extension access (consider restricting in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Customize this for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/frontend.html")


@app.get("/debug_transcript/{video_id}")
def debug_transcript(video_id: str):
    """
    Debug endpoint to fetch and return the raw transcript JSON for a video id,
    useful for testing if transcripts are reachable via proxy.
    """
    try:
        # Optionally pass proxies here if needed (use same as in qa_engine)
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id
            # If you want to enforce proxies explicitly, e.g.:
            # , proxies=qa_engine.proxies
        )
        return {"transcript": transcript}
    except TranscriptsDisabled:
        return {"error": "Transcripts are disabled for this video."}
    except NoTranscriptFound:
        return {"error": "No transcript found for this video."}
    except Exception as e:
        return {"error": f"Failed to fetch transcript: {str(e)}"}


@app.post('/api/ask')
async def ask(request: Request):
    """
    Endpoint to answer questions about the given YouTube video URL using the QA engine.
    Expects JSON body with keys:
        - video_url: str
        - question: str
    Returns JSON with {"answer": "<text answer>"} or HTTP 400 for missing data.
    """
    data = await request.json()
    video_url = data.get('video_url')
    question = data.get('question')
    if not video_url or not question:
        raise HTTPException(status_code=400, detail="Missing video_url or question")

    try:
        answer = qa.ask(video_url, question)
        return {"answer": answer}
    except Exception as e:
        # Log here if desired
        return {"error": f"Failed to get answer: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        reload=True,  # Set to False in production
    )
