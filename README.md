youtube-transcript-q-a

This project enables users to ask questions about YouTube video transcripts via a question-answering system. It consists of a Python backend for transcript processing and a Chrome extension for seamless interaction.

Features
Extracts and processes YouTube video transcripts

Question-answering engine for user queries

Chrome extension for easy access directly from video pages

Project Structure
text
youtube-qa-backend/
    main.py
    qa_engine.py
    requirements.txt
    .env
youtube-qa-extension/
    icon.png
    popup.html
    popup.js
    manifest.json
    style.css
Usage
Backend
Run the Python backend (main.py) to serve the QA model/API.

Frontend
Load youtube-qa-extension as an unpacked extension in Chrome for the popup interface.

Deployment
Deployed the backend to cloud platforms like Render.
The extension can be loaded locally or published to the Chrome Web Store.

License
MIT 

