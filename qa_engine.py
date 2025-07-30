import os
import re
import logging
import html
from typing import List, Optional, Tuple
import urllib3
import requests

# --- Disable urllib3 InsecureRequestWarning globally ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Patch requests to disable SSL verification globally ---
_old_request = requests.Session.request


def _request_no_ssl(self, *args, **kwargs):
    kwargs['verify'] = False  # Disable SSL certificate verification
    return _old_request(self, *args, **kwargs)


requests.Session.request = _request_no_ssl

# ----------------------------------

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from deep_translator import GoogleTranslator
from pytube import YouTube

# --- ENV & LOGGING ---
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not set! Please set in the environment or .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- ScraperAPI Proxy Settings ---
SCRAPERAPI_KEY = "840841f3a6e68c37a29881d961d5c481"
SCRAPERAPI_PROXY = f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"

proxies = {
    "http": SCRAPERAPI_PROXY,
    "https": SCRAPERAPI_PROXY,
}


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|shorts\/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("No valid video ID found in URL")


def translate_to_english(text: str) -> str:
    try:
        detected = GoogleTranslator(source="auto", target="en").detect(text[:160])
        if detected and detected.lower() != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated
    except Exception as e:
        logger.warning(f"Translation detection/translation failed: {e}")
    return text


def get_transcript_docs(video_id: str) -> Tuple[Optional[List[Document]], bool]:
    try:
        transcript_entries = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-IN", "hi"], proxies=proxies
        )
        logger.info(f"Fetched {len(transcript_entries)} transcript entries.")
        text = " ".join([d["text"] for d in transcript_entries])
        logger.info(f"Transcript snippet: {text[:200]}")
        text_en = translate_to_english(text)
        return [Document(page_content=text_en)], True
    except Exception as e:
        logger.warning(f"Failed to fetch transcript: {e}")
        return None, False


def get_video_title(youtube_url: str) -> Optional[str]:
    try:
        video_id = extract_video_id(youtube_url)
        clean_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            yt = YouTube(clean_url)
            return yt.title
        except Exception as e:
            logger.warning(f"Could not fetch video title with pytube: {e}")

        try:
            r = requests.get(clean_url, proxies=proxies, timeout=8, verify=False)
            if r.status_code == 200:
                m = re.search(r"<title>(.*?) - YouTube</title>", r.text)
                if m:
                    title = html.unescape(m.group(1)).strip()
                    return title
        except Exception as e2:
            logger.warning(f"Could not fetch/parse video title from HTML: {e2}")
    except Exception as e:
        logger.warning(f"Failed extracting video title: {e}")
    return None


def web_search_links(query: str) -> str:
    import urllib.parse

    q_url = urllib.parse.quote(query)
    return (
        f"Sorry, I couldn't answer from the transcript.\n"
        f"You can try searching the web:\n"
        f"- Google: https://www.google.com/search?q={q_url}\n"
        f"- DuckDuckGo: https://duckduckgo.com/?q={q_url}"
    )


class YouTubeConversationalQA:
    def __init__(self, model="gpt-3.5-turbo"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vectorstore_cache = {}
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model=model, temperature=0.4, max_tokens=512
        )
        self.convs = {}
        self.last_transcript_used = False  # Tracks transcript fetch status

    def build_chain(self, video_url: str, session_id: str = "default"):
        video_id = extract_video_id(video_url)
        if video_id not in self.vectorstore_cache:
            docs, transcript_ok = get_transcript_docs(video_id)
            self.last_transcript_used = transcript_ok
            if not docs:
                logger.warning(f"No transcript docs found for video_id={video_id}")
                return None
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            splits = splitter.split_documents(docs)
            vdb = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore_cache[video_id] = vdb
        retriever = self.vectorstore_cache[video_id].as_retriever()
        if session_id not in self.convs:
            self.convs[session_id] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.convs[session_id],
            return_source_documents=False,
        )

    # removed is_incomplete method entirely (no filtering)

    def ask(self, video_url: str, question: str, session_id: str = "default") -> Tuple[str, bool]:
        context_answer = None
        chain = self.build_chain(video_url, session_id)
        if chain is not None:
            try:
                result = chain.invoke({"question": question})
                context_answer = (result.get("answer", "") if result else "").strip()
            except Exception as e:
                logger.warning(f"Transcript-based QA failed: {e}")
                context_answer = None
        # Return transcript-based answer even if short or vague
        if context_answer:
            return context_answer, self.last_transcript_used

        # No transcript or empty answer; final fallback: provide web search links
        return web_search_links(question), False


if __name__ == "__main__":
    print("Welcome to YouTube Q&A! Paste any public YouTube video URL.")
    qa = YouTubeConversationalQA()
    url = input("YouTube video URL: ")
    session = "user1"
    print(
        "You can ask multiple questions about this video! (Press Enter on empty line to exit)\n"
    )
    while True:
        q = input("Your question (or just Enter to exit): ")
        if not q.strip():
            break
        try:
            ans, used_transcript = qa.ask(url, q, session)
            if used_transcript:
                print(f"\n[ANSWERED FROM TRANSCRIPT]\nAnswer: {ans}\n")
            else:
                print(f"\n[NOT FROM TRANSCRIPT, fallback used]\nAnswer: {ans}\n")
        except Exception as e:
            print(f"Error: {e}")
