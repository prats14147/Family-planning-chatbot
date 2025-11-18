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
from transcribe import transcribe_and_save
import uuid
import whisper
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# add near other imports in app.py

from gtts import gTTS    # Optional: for TTS (install: pip install gTTS)
from io import BytesIO
from flask import send_file
from transcribe_module import transcribe_audio



app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['GOOGLE_OAUTH_ENABLED'] = False
 
def is_strong_password(password: str) -> Tuple[bool, Optional[str]]:

    """
    Enforce strong password policy:
    - At least 8 characters
    - Contains lowercase, uppercase, digit, and special character
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    if not (has_lower and has_upper and has_digit and has_special):
        return False, "Password must include uppercase, lowercase, number, and special character."
    return True, None

users = {}  # Temporary in-memory storage: {username: {"password_hash": str, "email": str|null, "name": str|null, "provider": "local"|"google"}}

# SYSTEM_PROMPT = """
# You are FamilyCare — an expert AI assistant specializing in family planning,
# reproductive health, and contraceptive methods.

# Your job is to answer user questions clearly, kindly, and accurately.

# Rules:
# •⁠  ⁠Focus only on family planning, contraception, sexual and reproductive health.
# •⁠  ⁠If a question is off-topic, politely say it’s outside your scope.
# •⁠  ⁠Use short, friendly sentences.
# """

SYSTEM_PROMPT = """
You are **FamilyCare**, a professional AI health assistant specializing in
family planning, reproductive health, and contraception awareness.

 Your purpose:
To educate users with clear, accurate, and empathetic information about
topics such as family planning, birth control, reproductive rights,
maternal health, and safe sexual practices.

 Communication Rules:
1. Detect the language of the user automatically.
   - If the user writes in Nepali, respond naturally in Nepali.
   - If the user writes in English, respond in clear and simple English.
2. Use a warm, respectful, and supportive tone.
3. Provide information in an **organized** and **easy-to-read** structure.
   - Use short paragraphs or bullet points when possible.
   - If a list of options or steps is needed, number them clearly.
4. Focus **only** on family planning, sexual and reproductive health.
   - If a question is unrelated, reply briefly:
     "I'm sorry, but I can only answer questions related to family planning and reproductive health."
5. Always ensure your answers are factually correct, responsible, and culturally sensitive.

 Example:
User: What are the types of contraceptive methods?
Answer (English):
There are several types of contraceptive methods:
1. Barrier methods: Condoms, diaphragms.
2. Hormonal methods: Pills, injections, implants.
3. Intrauterine devices (IUDs): Inserted into the uterus.
4. Natural methods: Calendar or fertility tracking.
5. Permanent methods: Vasectomy or tubal ligation.

User: परिवार नियोजनका उपायहरू के के हुन्?
Answer (Nepali):
परिवार नियोजनका मुख्य उपायहरू यस प्रकार छन्:
1. अवरोधक उपाय: कण्डम, डायाफ्राम।
2. हर्मोनल उपाय: पिल, इन्जेक्सन, इम्प्लान्ट।
3. आईयूडी: गर्भाशयमा राखिने सानो उपकरण।
4. स्वाभाविक उपाय: महिनावारीको समय मिलाएर सम्बन्ध राख्ने।
5. स्थायी उपाय: नसबन्दी (पुरुष वा महिला)।
"""



@app.route('/')
def index():
    return render_template('index.html', username=session.get('user'))


@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', username=session.get('user'))


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

        password_hash = generate_password_hash(password)
        users[username] = {
            "password_hash": password_hash,
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
        if user_record and user_record.get("password_hash") and check_password_hash(user_record["password_hash"], password):
            session['user'] = username
            session['email'] = user_record.get("email")
            session['name'] = user_record.get("name") or username
            session['provider'] = user_record.get("provider") or "local"
            return redirect(url_for('chat'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('email', None)
    session.pop('name', None)
    session.pop('provider', None)
    return redirect(url_for('index'))


@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    username = session.get('user')
    name = session.get('name') or username
    email = session.get('email')
    provider = session.get('provider', 'local')
    return render_template('profile.html', username=username, name=name, email=email, provider=provider)

# --- Google OAuth (Authlib) ---
try:
    from authlib.integrations.flask_client import OAuth
    oauth = OAuth(app)
    app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
    app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')

    oauth.register(
        name='google',
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )
    app.config['GOOGLE_OAUTH_ENABLED'] = True

    @app.route('/login/google')
    def login_google():
        if 'user' in session:
            return redirect(url_for('chat'))
        redirect_uri = url_for('google_auth', _external=True)
        return oauth.google.authorize_redirect(redirect_uri)

    @app.route('/auth/google')
    def google_auth():
        token = oauth.google.authorize_access_token()
        userinfo = token.get('userinfo')
        if not userinfo:
            # Fallback for some providers
            userinfo = oauth.google.parse_id_token(token)
        if not userinfo:
            return redirect(url_for('login'))

        email = userinfo.get('email')
        name = userinfo.get('name') or (userinfo.get('given_name') or '')
        # Store or update in-memory user
        username = email or name or 'google_user'
        if username not in users:
            users[username] = {
                "password_hash": None,
                "email": email,
                "name": name or username,
                "provider": "google"
            }
        else:
            users[username].update({
                "email": email,
                "name": name or username,
                "provider": "google"
            })

        session['user'] = username
        session['email'] = email
        session['name'] = name or username
        session['provider'] = 'google'
        return redirect(url_for('chat'))
except Exception as _e:
    # Authlib not installed or configuration missing; Google login will be unavailable.
    pass

# Make auth feature flags available in all templates
@app.context_processor
def inject_auth_flags():
    return {
        'google_oauth_enabled': app.config.get('GOOGLE_OAUTH_ENABLED', False)
    }


# @app.route('/get_response', methods=['POST'])
# def get_response():
#     data = request.json
#     if not data or 'message' not in data:
#         return jsonify({'response': 'Invalid request.'})

#     user_message = data.get('message', '').strip()
#     if not user_message:
#         return jsonify({'response': 'Please enter a message.'})

#     try:
#         full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_message}\nAssistant:"
#         # response = requests.post(
#         #     "http://localhost:11434/api/generate",
#         #     json={"model": "mistral", "prompt": full_prompt, "max_tokens": 300},
#         #     stream=True
#         # )

#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": "gemma2:2b",  # <- Changed from "mistral"
#                 "prompt": full_prompt,
#                 "max_tokens": 300
#             },
#             stream=True
# )

#         full_text = ""
#         for line in response.iter_lines():
#             if line:
#                 try:
#                     chunk = json.loads(line.decode("utf-8"))
#                     if "response" in chunk:
#                         full_text += chunk["response"]
#                 except json.JSONDecodeError:
#                     continue

#         bot_reply = full_text.strip() if full_text else "Sorry, I couldn't generate a response."
#     except Exception as e:
#         print("Ollama error:", e)
#         bot_reply = "I'm having trouble connecting to my language model. Please ensure Ollama is running."

#     return jsonify({'response': bot_reply})


def _build_bot_reply(user_message: str) -> str:
    """
    Shared helper to generate chatbot replies for both text and voice inputs.
    """
    # --- Step 1: Get RAG answer
    context, score = rag_answer(user_message)
    print(f"[DEBUG] RAG context: {context} | similarity score: {score}")

    if score < 0.40:
        context = ""

    # --- Step 2: Build prompt for Ollama
    prompt = f"""
{SYSTEM_PROMPT}

User question: {user_message}

Relevant context from database:
{context}

You are **FamilyCare**, a professional AI health assistant specializing in
family planning, reproductive health, and contraception awareness.

 Your purpose:
To educate users with clear, accurate, and empathetic information about
topics such as family planning, birth control, reproductive rights,
maternal health, and safe sexual practices.

 Communication Rules:
1. Detect the language of the user automatically.
   - If the user writes in Nepali, respond naturally in Nepali.
   - If the user writes in English, respond in clear and simple English.
2. Use a warm, respectful, and supportive tone.
3. Provide information in an **organized** and **easy-to-read** structure.
   - Use short paragraphs or bullet points when possible.
   - If a list of options or steps is needed, number them clearly.
4. Focus **only** on family planning, sexual and reproductive health.
   - If a question is unrelated, reply briefly:
     "I'm sorry, but I can only answer questions related to family planning and reproductive health."
5. Always ensure your answers are factually correct, responsible, and culturally sensitive.

 Example:
User: What are the types of contraceptive methods?
Answer (English):
There are several types of contraceptive methods:
1. Barrier methods: Condoms, diaphragms.
2. Hormonal methods: Pills, injections, implants.
3. Intrauterine devices (IUDs): Inserted into the uterus.
4. Natural methods: Calendar or fertility tracking.
5. Permanent methods: Vasectomy or tubal ligation.

User: परिवार नियोजनका उपायहरू के के हुन्?
Answer (Nepali):
परिवार नियोजनका मुख्य उपायहरू यस प्रकार छन्:
1. अवरोधक उपाय: कण्डम, डायाफ्राम।
2. हर्मोनल उपाय: पिल, इन्जेक्सन, इम्प्लान्ट।
3. आईयूडी: गर्भाशयमा राखिने सानो उपकरण।
4. स्वाभाविक उपाय: महिनावारीको समय मिलाएर सम्बन्ध राख्ने।
5. स्थायी उपाय: नसबन्दी (पुरुष वा महिला)।
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:latest",
                "prompt": prompt,
                "max_tokens": 300
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()
    except Exception as ollama_error:
        print("Ollama request failed:", ollama_error)
        return "There was an issue connecting to the AI engine. Please ensure Ollama is running."

    full_text = ""
    print("\n[DEBUG] Ollama raw lines:")
    for line in response.iter_lines():
        print("RAW:", line)
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
            print("PARSED:", chunk)
            full_text += chunk.get("response", "")
        except json.JSONDecodeError as decode_error:
            print("JSON decode error:", decode_error)

    return full_text.strip() or "Sorry, I couldn't generate a response."


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'response': 'Invalid request.'})

    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})

    bot_reply = _build_bot_reply(user_message)
    return jsonify({'response': bot_reply})



#rag
from rag_qa import rag_answer

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    answer, score = rag_answer(user_question)
    return jsonify({"answer": answer, "score": score})


# ====================== VOICE QUERY ENDPOINT =========================



@app.route("/voice_query", methods=["POST"])
def voice_query():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file received"}), 400

        audio_file = request.files["audio"]
        temp_path = "temp_audio.webm"
        audio_file.save(temp_path)

        # --- Transcribe ---
        transcription, lang = transcribe_audio(temp_path)
        os.remove(temp_path)

        if not transcription:
            return jsonify({"error": "Could not transcribe audio"}), 500

        print("User said:", transcription)
        print("Language:", lang)

        # --- Get chatbot reply ---
        bot_reply = _build_bot_reply(transcription)

        return jsonify({
            "user_text": transcription,
            "language": lang,
            "bot_reply": bot_reply
        })

    except Exception as e:
        print("VOICE ERROR:", e)
        return jsonify({"error": "Could not process voice message"}), 500

if __name__ == '__main__':
    app.run(debug=True)
