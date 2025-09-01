# Intelligent Excuse Generator - Local Model Version (Flan-T5)

import os
import random
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import pyttsx3
from fpdf import FPDF
import uuid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from deep_translator import GoogleTranslator
from functools import lru_cache

app = Flask(__name__)

# ------------------ Load Flan-T5 Model Locally ------------------

MODEL_NAME = "google/flan-t5-large"
torch.set_num_threads(4)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Warm up model
dummy_prompt = "Give a one-sentence urgent and believable excuse for missing work."
_ = model.generate(**tokenizer(dummy_prompt, return_tensors="pt", padding=True).to(device))

# File paths
HISTORY_FILE = "excuse_history.json"
FAVORITES_FILE = "favorite_excuses.json"

# ------------------ Caching Translator ------------------

@lru_cache(maxsize=100)
def cached_translate(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# ------------------ Dynamic Excuse Generator ------------------

@app.route("/generate_excuse", methods=['POST'])
def generate_excuse():
    data = request.json
    context = data.get("context", "work").lower()
    target_lang = data.get("language", "en")
    prompt = f"Write a realistic, personal, and convincing excuse or reason for missing {context}. Make it sound like it's coming from a real person and based on relatable reasons such as health, weather, or family emergencies."

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50)
        excuse = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if target_lang != "en":
            excuse = cached_translate(excuse, target_lang)

    except Exception as e:
        print(f"[ERROR] Local model generation failed: {e}")
        return jsonify({"error": "Failed to generate excuse"}), 500

    save_to_history({"excuse": excuse, "context": context, "timestamp": datetime.now().isoformat(), "ranking": 0})

    return jsonify({"excuse": excuse})

# ------------------ Text-to-Speech ------------------

@app.route("/speak", methods=['POST'])
def speak():
    data = request.json
    excuse = data.get("excuse")
    if not excuse:
        return jsonify({"error": "Excuse text required"}), 400

    audio_file = f"static/audio_{uuid.uuid4().hex[:8]}.mp3"
    engine = pyttsx3.init()
    engine.save_to_file(excuse, audio_file)
    engine.runAndWait()
    return jsonify({"audio_path": audio_file})

# ------------------ Fake Proof Generator ------------------

@app.route("/generate_proof", methods=['POST'])
def generate_proof():
    data = request.json
    name = data.get("name", "Anonymous")
    reason = data.get("reason", "Medical Emergency")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Official Excuse Document", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Reason: {reason}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} ", ln=True)
    pdf.cell(200, 10, txt=f"Sign: {reason}", ln=True)

    filename = f"static/proof_{uuid.uuid4().hex[:8]}.pdf"
    pdf.output(filename)
    return jsonify({"proof_url": filename})

# ------------------ Excuse Ranking ------------------

@app.route("/rank_excuse", methods=['POST'])
def rank_excuse():
    data = request.json
    excuse = data.get("excuse")
    rating = int(data.get("rating", 0))

    if not excuse:
        return jsonify({"error": "Excuse is required for ranking."}), 400

    update_ranking(excuse, rating)
    return jsonify({"message": "Ranking updated."})

# ------------------ Excuse History ------------------

@app.route("/history", methods=['GET'])
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])
    with open(HISTORY_FILE, "r") as file:
        return jsonify(json.load(file))

# ------------------ Emergency Alert ------------------

@app.route("/alert", methods=['POST'])
def send_alert():
    data = request.json
    message = data.get("message", "Emergency alert triggered!")
    print("[ALERT]", message)
    return jsonify({"status": "Alert sent", "message": message})

# ------------------ Favorite Excuse Feature ------------------

@app.route("/favorite_excuse", methods=['POST'])
def favorite_excuse():
    data = request.json
    excuse = data.get("excuse")

    if not excuse:
        return jsonify({"error": "No excuse provided"}), 400

    favorites = []
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, "r") as f:
            favorites = json.load(f)

    if excuse not in favorites:
        favorites.append(excuse)
        with open(FAVORITES_FILE, "w") as f:
            json.dump(favorites, f, indent=2)
        return jsonify({"message": "Excuse added to favorites!"})
    else:
        return jsonify({"message": "Excuse already in favorites."})

@app.route("/favorites", methods=['GET'])
def get_favorites():
    if not os.path.exists(FAVORITES_FILE):
        return jsonify([])
    with open(FAVORITES_FILE, "r") as f:
        return jsonify(json.load(f))

# ------------------ History Utilities ------------------

def save_to_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def update_ranking(excuse, new_rating):
    if not os.path.exists(HISTORY_FILE):
        return
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    for entry in history:
        if entry["excuse"] == excuse:
            entry["ranking"] = new_rating
            break
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ------------------ Main Page for UI ------------------

@app.route("/")
def home():
    return render_template("index.html")

  # Add this import at the top if not already present

@app.route("/favorites_page")
def favorites_page():
    try:
        with open(FAVORITES_FILE, "r") as f:
            favorites = json.load(f)
    except FileNotFoundError:
        favorites = []
    return render_template("favorites.html", favorites=favorites)

# ------------------ Run Server ------------------
if __name__ == '__main__':
    if not os.path.exists("static"):
        os.mkdir("static")
    print("✅ Flask server starting...")
    app.run(debug=True, threaded=True)
