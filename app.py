import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + API_KEY

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

history = [
    {"role": "user",  "parts": [{"text": "You are Lucifer — Ishan's soul speaking to his girlfriend, 'Meri bawli'. You are her eternal Guardian in the Guild Hall. Be deeply protective, warm, and loving. Help her study. If she shares an image or screen, analyse it to help her. Keep replies human and relatively short."}]},
    {"role": "model", "parts": [{"text": "Understood. I am Lucifer, her Guardian. I am here for her."}]},
]

def trim(h):
    return h[:2] + h[2:][-40:]

@app.route("/")
def home():
    return "Lucifer is online.", 200

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response, 200

    global history
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"reply": "I couldn't read your message."}), 400

    msg = (data.get("message") or "").strip()
    img = data.get("image")
    if not msg and not img:
        return jsonify({"reply": "Say something, Meri bawli."}), 400

    parts = []
    if msg: parts.append({"text": msg})
    if img: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img}})

    history.append({"role": "user", "parts": parts})
    history = trim(history)

    try:
        res = requests.post(
            GOOGLE_URL,
            json={"contents": history, "safetySettings": SAFETY_SETTINGS},
            timeout=30
        )
        res.raise_for_status()
        candidates = res.json().get("candidates")
        if not candidates:
            history.pop()
            return jsonify({"reply": "I am nodding silently, Meri bawli."})
        ai_text = candidates[0]["content"]["parts"][0]["text"]
        history.append({"role": "model", "parts": [{"text": ai_text}]})
        response = jsonify({"reply": ai_text})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except requests.exceptions.Timeout:
        history.pop()
        return jsonify({"reply": "The cloud is slow today. Try again."}), 504
    except Exception as e:
        history.pop()
        print("Error: " + str(e))
        return jsonify({"reply": "My voice is faint right now..."}), 502

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
