import urllib.request
import urllib.parse
import re
from flask import Flask, render_template, request, jsonify, Response
from prompt import translate_text


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json() or {}
    text = data.get("text", "")
    source_lang = data.get("source_lang", "Détecter la langue")
    target_lang = data.get("target_lang", "Anglais")
    input_chars = data.get("input_chars")
    
    if not text.strip():
        return jsonify({"translated_text": ""})
    
    # Return translation text from prompt.py
    translated = translate_text(text, source_lang, target_lang, input_chars)
    return jsonify({"translated_text": translated})

@app.route("/api/tts")
def tts():
    text = request.args.get("text", "")
    lang = request.args.get("lang", "en")
    
    if not text.strip():
        return Response(status=400)
    
    # Simplify language code (e.g. en-US -> en)
    special_cases = {
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "pt-br": "pt-BR",
        "pt-pt": "pt-PT",
    }
    lang_lower = lang.lower()
    clean_lang = special_cases.get(lang_lower, lang.split('-')[0])
    
    # Split text into chunks under 150 characters to prevent Google's limit errors
    # We split by punctuation/spaces to keep natural pauses, and slice long strings if needed.
    max_len = 140
    parts = re.split(r'([.?!,;。？！，、\s]+)', text)
    chunks = []
    current_chunk = ""
    for part in parts:
        if not part:
            continue
        if len(current_chunk) + len(part) < max_len:
            current_chunk += part
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # If a single part exceeds max_len, slice it to prevent data loss
            if len(part) >= max_len:
                for i in range(0, len(part), max_len):
                    sub_part = part[i:i+max_len].strip()
                    if sub_part:
                        chunks.append(sub_part)
            else:
                current_chunk = part
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    combined_audio = b""
    try:
        for chunk in chunks:
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={clean_lang}&client=tw-ob&q={urllib.parse.quote(chunk)}"
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': 'https://translate.google.com/'
                }
            )
            # Added 10-second timeout to prevent indefinite blocking
            with urllib.request.urlopen(req, timeout=10) as response:
                combined_audio += response.read()
                
        return Response(combined_audio, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=54444, use_reloader=False)
