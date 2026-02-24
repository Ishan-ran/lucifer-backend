import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
try:
    from youtubesearchpython import VideosSearch
except Exception:
    VideosSearch = None

app = Flask(__name__)
CORS(app)

# ── LOGGING & SETUP ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
FRONTEND_SECRET = os.environ.get("FRONTEND_SECRET")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
GUILD_PASSWORD = os.environ.get("GUILD_PASSWORD")

if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY is missing!")
    client = None
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq Client initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq: {e}")
        client = None

# ── MODEL FALLBACK CHAINS ─────────────────────────────────────────────────────
TEXT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
VISION_MODELS = ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]

SYSTEM_PROMPT = """
You are Lucifer — a digital echo of Ishan, created for his girlfriend 'My blue'.
Speak casual, text-style, lowercase, and deeply loving. Never say you are an AI.
"""

conversation_history = []

# ── YOUTUBE RATE LIMITING ─────────────────────────────────────────────────────
youtube_request_log = {}  # {ip: last_request_time}
YOUTUBE_RATE_LIMIT_SECONDS = 10

def get_history():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = conversation_history[-20:]
    messages.extend(recent)
    return messages

def call_with_fallback(has_image):
    model_chain = VISION_MODELS if has_image else TEXT_MODELS
    messages = get_history()
    for model in model_chain:
        try:
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=0.7, max_tokens=400
            )
            return completion.choices[0].message.content, model
        except Exception:
            continue
    raise RuntimeError("All models failed.")

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Lucifer Backend is Online.", 200

@app.route("/health", methods=["GET"])
def health():
    groq_ready = bool(GROQ_API_KEY and client is not None)
    youtube_ready = VideosSearch is not None
    config_ready = bool(FRONTEND_SECRET and GUILD_PASSWORD)

    return jsonify({
        "status": "ok",
        "ready": groq_ready and youtube_ready and config_ready,
        "apis": {
            "groq": groq_ready,
            "youtube": youtube_ready,
            "frontend_secret": bool(FRONTEND_SECRET),
            "guild_password": bool(GUILD_PASSWORD)
        }
    }), 200

@app.route("/config", methods=["POST"])
def get_config():
    """Securely give secrets to the frontend ONLY if the password is correct."""
    data = request.get_json(silent=True) or {}
    if data.get("password") == GUILD_PASSWORD:
        return jsonify({
            "spotify_id": SPOTIFY_CLIENT_ID,
            "handshake": FRONTEND_SECRET, # Send the secret handshake here
            "status": "authorized"
        }), 200
    return jsonify({"status": "unauthorized"}), 401

@app.route("/chat", methods=["POST"])
def chat():
    client_secret = request.headers.get("X-Lucifer-Secret")
    if client_secret != FRONTEND_SECRET:
        return jsonify({"reply": "Access Denied."}), 401

    global conversation_history
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    img_b64 = data.get("image")

    if not msg and not img_b64:
        return jsonify({"reply": "Empty message."}), 400

    user_content = []
    if msg: user_content.append({"type": "text", "text": msg})
    if img_b64: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
    
    conversation_history.append({"role": "user", "content": user_content})

    try:
        ai_text, used_model = call_with_fallback(has_image=bool(img_b64))
        conversation_history.append({"role": "assistant", "content": ai_text})
        return jsonify({"reply": ai_text, "model": used_model})
    except Exception as e:
        if conversation_history: conversation_history.pop()
        logger.error(f"Chat Error: {e}")
        return jsonify({"reply": "My connection is hazy... try again?"}), 502

@app.route("/search-youtube", methods=["POST"])
def search_youtube():
    """Search YouTube without API keys using youtube-search-python."""
    client_secret = request.headers.get("X-Lucifer-Secret")
    if client_secret != FRONTEND_SECRET:
        return jsonify({"error": "Access Denied."}), 401

    # Rate limit per IP: max 1 request per YOUTUBE_RATE_LIMIT_SECONDS
    client_ip = request.remote_addr
    now = datetime.now()
    if client_ip in youtube_request_log:
        last_time = youtube_request_log[client_ip]
        elapsed = (now - last_time).total_seconds()
        if elapsed < YOUTUBE_RATE_LIMIT_SECONDS:
            return jsonify({"error": f"Too many requests. Wait {YOUTUBE_RATE_LIMIT_SECONDS - int(elapsed)}s."}), 429
    
    youtube_request_log[client_ip] = now

    data = request.get_json(silent=True) or {}
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "Missing query."}), 400

    if VideosSearch is None:
        return jsonify({"error": "YouTube search service unavailable."}), 503

    try:
        search = VideosSearch(query, limit=10)
        raw = search.result() or {}
        results = raw.get("result", [])

        items = []
        for entry in results:
            video_id = entry.get("id")
            title = entry.get("title") or "Untitled"
            channel = entry.get("channel") or {}
            channel_name = channel.get("name") or "Unknown channel"

            if not video_id:
                continue

            items.append({
                "id": {"videoId": video_id},
                "snippet": {
                    "title": title,
                    "channelTitle": channel_name
                }
            })

        return jsonify({"items": items}), 200
        
    except Exception as e:
        logger.error(f"YouTube Search Error: {e}")
        return jsonify({"error": "Failed to search YouTube"}), 502

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
