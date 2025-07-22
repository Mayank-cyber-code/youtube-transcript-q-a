const BACKEND_API_URL = "https://youtube-q-a.onrender.com/api/ask"; // Change if your backend URL differs

function displayResult(msg, disabled = false) {
    document.getElementById('result').textContent = msg;
    document.getElementById('ask_btn').disabled = disabled;
    document.getElementById('question').disabled = disabled;
}

document.addEventListener("DOMContentLoaded", function () {
    chrome.tabs.query({ active: true, lastFocusedWindow: true }, function (tabs) {
        const tab = tabs[0];
        if (tab && tab.url && tab.url.match(/^https:\/\/www\.youtube\.com\/watch\?v=/)) {
            document.getElementById('video_url_group').style.display = "none";
            // Do NOT set question value or auto-fetch
            document.getElementById('ask_btn').disabled = false;
            document.getElementById('question').disabled = false;
            // Store video URL for question send
            document.getElementById('ask_btn').dataset.videoUrl = tab.url;
        } else {
            document.getElementById('video_url_group').style.display = "none";
            displayResult('Please load a YouTube "watch" video in your current tab and reload the extension.', true);
        }
    });

    document.getElementById('ask_btn').addEventListener('click', () => {
        const video_url = document.getElementById('ask_btn').dataset.videoUrl;
        const question = document.getElementById('question').value.trim();
        if (!video_url) {
            displayResult("Please open a YouTube video and reload the extension.", true);
            return;
        }
        if (!question) {
            displayResult("Please enter a question.");
            return;
        }
        fetchAnswer(video_url, question);
    });
});

function fetchAnswer(video_url, question) {
    displayResult('Loading...', true);
    fetch(BACKEND_API_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({video_url, question})
    })
    .then(res => res.json())
    .then(data => {
        if(data.answer) {
            displayResult(data.answer);
        } else if(data.error) {
            displayResult("Error: " + data.error);
        } else {
            displayResult("No answer returned.");
        }
        document.getElementById('ask_btn').disabled = false;
        document.getElementById('question').disabled = false;
    })
    .catch(err => {
        displayResult("Error: " + err, true);
    });
}

