from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import requests
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple
import uuid

# from transcribe import transcribe_and_save
# import whisper
# import torch
# import torchaudio
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from transcribe_module import transcribe_audio

from gtts import gTTS
from io import BytesIO
from flask import send_file
from rag_qa import rag_answer

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['GOOGLE_OAUTH_ENABLED'] = False

# ---------------- PASSWORD POLICY --------------------

def is_strong_password(password: str) -> Tuple[bool, Optional[str]]:
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    if not (has_lower and has_upper and has_digit and has_special):
        return False, "Password must include uppercase, lowercase, number, and special character."
    return True, None

# ---------------- USERS MEMORY --------------------

users = {}

# ---------------- SYSTEM PROMPT --------------------

SYSTEM_PROMPT = """(same — unchanged for brevity)"""

# ---------------- BASIC ROUTES --------------------

@app.route('/')
def index():
    return render_template('index.html', username=session.get('user'))

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', username=session.get('user'))

# ---------------- SIGNUP / LOGIN --------------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user' in session:
        return redirect(url_for('chat'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('signup.html', error="Username already exists!")
        ok, msg = is_strong_password(password)
        if not ok:
            return render_template('signup.html', error=msg)
        users[username] = {
            "password_hash": generate_password_hash(password),
            "email": None,
            "name": username,
            "provider": "local"
        }
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('chat'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_record = users.get(username)
        if user_record and check_password_hash(user_record["password_hash"], password):
            session['user'] = username
            session['email'] = user_record.get("email")
            session['name'] = user_record.get("name")
            session['provider'] = "local"
            return redirect(url_for('chat'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ---------------- CHATBOT SAFE MODE --------------------

def _build_bot_reply(user_message: str) -> str:
    # ---- RAG call safe ----
    try:
        context, score = rag_answer(user_message)
    except:
        context = ""
        score = 0

    if score < 0.4:
        context = ""

    prompt = f"{SYSTEM_PROMPT}\n\nUser message:\n{user_message}\n\nContext:\n{context}"

    # ---- OLLAMA BLOCKED ON RENDER ----
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral:latest", "prompt": prompt, "max_tokens": 300},
            stream=True,
            timeout=120
        )
        response.raise_for_status()
    except:
        return "Model not available in cloud deployment."

    # (Won't run on Render anyway but kept safe)
    full_text = ""
    try:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode())
                full_text += chunk.get("response", "")
            except:
                pass
    except:
        return "Model not available in cloud deployment."

    return full_text.strip()

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    msg = data.get('message', '').strip()
    if not msg:
        return jsonify({'response': 'Please enter a message.'})
    reply = _build_bot_reply(msg)
    return jsonify({'response': reply})

# ------------------- RAG ---------------------

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question")
    if not q:
        return jsonify({"error": "No question provided"}), 400
    try:
        a, score = rag_answer(q)
    except:
        return jsonify({"answer": "RAG not available in cloud deployment", "score": 0})
    return jsonify({"answer": a, "score": score})


# ------------------- RENDER RUN ---------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
