import os
import re
import logging
from typing import List, Optional

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
import wikipedia
from pytube import YouTube
import requests
import html

# --- ENV & LOGGING ---
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not set! Please set in the environment or .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Proxy Settings for YouTubeTranscriptApi ---
TOR_PROXY_IP = "16.170.237.7"  # Replace with your EC2 public IP
TOR_PROXY_PORT = 9050
proxies = {
        "http": "socks5://16.170.237.7:9050",
        "https": "socks5://16.170.237.7:9050",
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


def get_transcript_docs(video_id: str) -> Optional[List[Document]]:
    """
    Fetch transcript using get_transcript() call via Tor proxy.
    Falls back to Wikipedia if transcript is not available.
    """
    try:
        transcript_entries = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-IN", "hi"], proxies=PROXIES
        )
        logger.info(f"Fetched transcript with {len(transcript_entries)} entries")
        text = " ".join([d["text"] for d in transcript_entries])
        text_en = translate_to_english(text)
        return [Document(page_content=text_en)]
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.warning(f"Transcript could not be fetched: {e}. Falling back to Wikipedia.")
    except Exception as e:
        logger.error(f"Error loading transcript: {e}")
    return None


def get_video_title(youtube_url: str) -> Optional[str]:
    try:
        video_id = extract_video_id(youtube_url)
        clean_url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(clean_url)
        return yt.title
    except Exception as e:
        logger.warning(f"Could not fetch video title with pytube: {e}")
        # Try HTML parse fallback
        try:
            page_url = f"https://www.youtube.com/watch?v={extract_video_id(youtube_url)}"
            r = requests.get(page_url, timeout=8)
            if r.status_code == 200:
                m = re.search(r"<title>(.*?) - YouTube</title>", r.text)
                if m:
                    title = html.unescape(m.group(1)).strip()
                    return title
        except Exception as e2:
            logger.warning(f"Could not fetch/parse video title from HTML: {e2}")
    return None


def clean_video_title_for_wikipedia(title: str) -> str:
    for sep in ["|", "-"]:
        if sep in title:
            title = title.split(sep)[0]
    return title.strip()


def wikipedia_search(query: str) -> Optional[str]:
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"According to Wikipedia:\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            sub_summary = wikipedia.summary(e.options[0], sentences=2)
            return f"According to Wikipedia ({e.options[0]}):\n{sub_summary}"
        except Exception:
            return None
    except wikipedia.exceptions.PageError:
        return None
    except Exception as ex:
        logger.warning(f"Wikipedia search error: {ex}")
        return None


def web_search_links(query: str) -> str:
    import urllib.parse

    q_url = urllib.parse.quote(query)
    return (
        f"Sorry, I couldn't answer from the transcript or Wikipedia.\n"
        f"You can try searching the web:\n"
        f"- [Google](https://www.google.com/search?q={q_url})\n"
        f"- [DuckDuckGo](https://duckduckgo.com/?q={q_url})"
    )


def clean_for_wikipedia(query: str) -> str:
    query = query.strip()
    match = re.match(
        r"(who|what|when|where|why|how)\s+(is|are|was|were|do|does|did|has|have|can|could|should|would)?\s*(.*)",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        topic = match.group(3).strip(" .?")
        return topic
    return query


VAGUE_PATTERNS = [
    "do not like each other",
    "i don't know",
    "i do not know",
    "not mentioned",
    "not provided",
    "not stated",
    "no idea",
    "no information",
    "no details",
    "insufficient information",
    "unclear",
    "unable to determine",
    "cannot determine",
    "can't say",
    "no context",
    "context not found",
    "the transcript does not",
    "sorry",
    "unfortunately",
]


def is_summary_question(question: str) -> bool:
    qs = question.lower()
    return (
        "what is this video about" in qs
        or "what is the topic" in qs
        or "main topic" in qs
        or "summarize" in qs
        or "summary" in qs
    )


class YouTubeConversationalQA:
    def __init__(self, model="gpt-3.5-turbo"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vectorstore_cache = {}
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model=model, temperature=0.4, max_tokens=512
        )
        self.convs = {}

    def build_chain(self, video_url: str, session_id: str = "default"):
        video_id = extract_video_id(video_url)
        if video_id not in self.vectorstore_cache:
            docs = get_transcript_docs(video_id)
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

    def is_incomplete(self, text: str) -> bool:
        if not text or len(text.strip()) < 8:
            return True
        lowered = text.lower()
        for pat in VAGUE_PATTERNS:
            if pat in lowered:
                return True
        return False

    def ask(self, video_url: str, question: str, session_id: str = "default") -> str:
        fallback_to_title = False
        context_answer = None
        chain = self.build_chain(video_url, session_id)
        if chain is not None:
            try:
                if is_summary_question(question):
                    custom_template = (
                        "Given the following video transcript, briefly summarize the main topic or content "
                        "of this video. Only use context, do NOT speculate. Transcript: {context}\n"
                        "In English, answer in 2-4 sentences."
                    )
                    result = chain.invoke({"question": custom_template})
                else:
                    result = chain.invoke({"question": question})
                context_answer = (result.get("answer", "") if result else "").strip()
            except Exception as e:
                logger.warning(f"Transcript-based QA failed: {e}")
                context_answer = None
                fallback_to_title = True
        else:
            fallback_to_title = True

        if context_answer and not self.is_incomplete(context_answer):
            return context_answer

        title_q = None
        if fallback_to_title:
            title_q = get_video_title(video_url)
            search_term = title_q if title_q else question
            wiki_ans = wikipedia_search(search_term)

            if (not wiki_ans or self.is_incomplete(wiki_ans)) and title_q:
                short_search = clean_video_title_for_wikipedia(title_q)
                if short_search != search_term:
                    wiki_ans = wikipedia_search(short_search)

            if wiki_ans and not self.is_incomplete(wiki_ans):
                return wiki_ans

        wiki_ans = wikipedia_search(question)
        if wiki_ans and not self.is_incomplete(wiki_ans):
            return wiki_ans

        topic = clean_for_wikipedia(question)
        if topic != question:
            wiki_ans2 = wikipedia_search(topic)
            if wiki_ans2 and not self.is_incomplete(wiki_ans2):
                return wiki_ans2

        return web_search_links(question)


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
            ans = qa.ask(url, q, session)
            print(f"\nAnswer: {ans}\n")
        except Exception as e:
            print(f"Error: {e}")
