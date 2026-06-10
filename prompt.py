import os
import time
import threading
import math
from collections import OrderedDict
from mlx_lm import load, stream_generate

# Mapping from French names (as selected in UI) to standard ISO 639-1 / regionalized codes.
# Only including the 49 supported languages that match the model's core capabilities.
LANGUAGE_MAP = {
    "Afrikaans": "af",
    "Allemand": "de-DE",
    "Anglais": "en",
    "Arabe": "ar",
    "Bengali": "bn",
    "Bulgare": "bg",
    "Chinois": "zh",
    "Coréen": "ko",
    "Croate": "hr",
    "Danois": "da",
    "Espagnol": "es-ES",
    "Estonien": "et",
    "Finnois": "fi",
    "Français": "fr-FR",
    "Grec": "el",
    "Gujarati": "gu",
    "Hébreu": "he",
    "Hindi": "hi",
    "Hongrois": "hu",
    "Indonésien": "id",
    "Islandais": "is",
    "Italien": "it-IT",
    "Japonais": "ja",
    "Kannada": "kn",
    "Letton": "lv",
    "Lituanien": "lt",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Néerlandais": "nl",
    "Ourdou": "ur",
    "Persan": "fa",
    "Polonais": "pl",
    "Portugais": "pt-PT",
    "Pendjabi": "pa",
    "Roumain": "ro",
    "Russe": "ru",
    "Serbe": "sr",
    "Slovaque": "sk",
    "Slovène": "sl",
    "Suédois": "sv",
    "Tagalog": "fil-PH",
    "Tamoul": "ta",
    "Tchèque": "cs-CZ",
    "Télougou": "te",
    "Thaï": "th",
    "Turc": "tr",
    "Ukrainien": "uk",
    "Vietnamien": "vi",
    "Zoulou": "zu"
}

# ──────────────────────────────────────────────
# Global MLX model + tokenizer — loaded ONCE at import time
# ──────────────────────────────────────────────
print("\n--- Preloading TranslateGemma MLX model ---")
_load_start = time.time()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_mlx")
if os.path.exists(model_path) and os.listdir(model_path):
    _model_to_load = model_path
    print(f"Loading local MLX model from: {model_path}")
else:
    _model_to_load = "mlx-community/translategemma-4b-it-4bit"
    print(f"Loading model '{_model_to_load}' from Hugging Face Hub")

MODEL, TOKENIZER = load(_model_to_load)
print(f"Model loaded in {time.time() - _load_start:.1f}s — ready for inference!\n")

_GENERATE_LOCK = threading.Lock()
_KEEPALIVE_SECONDS = int(os.environ.get("TRANSLATE_KEEPALIVE_SECONDS", "900"))
_MIN_NEW_TOKENS = int(os.environ.get("TRANSLATE_MIN_NEW_TOKENS", "16"))
_MAX_NEW_TOKENS = int(os.environ.get("TRANSLATE_MAX_NEW_TOKENS", "160"))
_CACHE_SIZE = int(os.environ.get("TRANSLATE_CACHE_SIZE", "128"))
_CACHE_LOCK = threading.Lock()
_TRANSLATION_CACHE = OrderedDict()


def _max_tokens_for(text: str, client_chars: int | None) -> tuple[int, int, int]:
    server_chars = len(text)
    if client_chars != server_chars:
        client_chars = server_chars

    text_tokens = len(TOKENIZER.encode(text))
    estimated_tokens = math.ceil(text_tokens * 1.6) + 8
    max_new_tokens = min(_MAX_NEW_TOKENS, max(_MIN_NEW_TOKENS, estimated_tokens))
    return max_new_tokens, text_tokens, client_chars


def _base_lang(lang_code: str) -> str:
    return lang_code.split("-")[0].lower()


def _target_code(target_lang: str) -> str:
    return LANGUAGE_MAP.get(target_lang, "en")


def _source_code(text: str, source_lang: str) -> str:
    if source_lang and source_lang != "Détecter la langue":
        return LANGUAGE_MAP.get(source_lang, "fr-FR")

    try:
        from langdetect import detect
        return detect(text)
    except Exception as e:
        print(f"Language detection failed: {e}. Defaulting to 'fr'")
        return "fr"


def _cache_get(key):
    if _CACHE_SIZE <= 0:
        return None
    with _CACHE_LOCK:
        value = _TRANSLATION_CACHE.get(key)
        if value is not None:
            _TRANSLATION_CACHE.move_to_end(key)
        return value


def _cache_set(key, value: str) -> None:
    if _CACHE_SIZE <= 0:
        return
    with _CACHE_LOCK:
        _TRANSLATION_CACHE[key] = value
        _TRANSLATION_CACHE.move_to_end(key)
        while len(_TRANSLATION_CACHE) > _CACHE_SIZE:
            _TRANSLATION_CACHE.popitem(last=False)


def _build_prompt(text: str, source_code: str, target_code: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": text
                }
            ]
        }
    ]
    return TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )


def _token_count(text: str) -> int:
    try:
        return len(TOKENIZER.encode(text))
    except Exception:
        return 0


def _generate_translation(prompt: str, max_tokens: int) -> tuple[str, dict]:
    text_parts = []
    last_response = None
    finish_reason = None

    for response in stream_generate(
        MODEL,
        TOKENIZER,
        prompt=prompt,
        max_tokens=max_tokens
    ):
        text_parts.append(response.text)
        last_response = response
        if "<end_of_turn>" in "".join(text_parts):
            finish_reason = "stop"
            break

    response_text = "".join(text_parts).split("<end_of_turn>")[0].strip()
    stats = {
        "prompt_tokens": getattr(last_response, "prompt_tokens", 0),
        "prompt_tps": getattr(last_response, "prompt_tps", 0.0),
        "generation_tokens": getattr(last_response, "generation_tokens", 0),
        "generation_tps": getattr(last_response, "generation_tps", 0.0),
        "peak_memory": getattr(last_response, "peak_memory", 0.0),
        "finish_reason": finish_reason or getattr(last_response, "finish_reason", None),
    }
    return response_text, stats


def _warm_model() -> None:
    try:
        prompt = _build_prompt("Bonjour", "fr-FR", "en")
        with _GENERATE_LOCK:
            _generate_translation(prompt, 16)
        print("[translate] Warmup complete")
    except Exception as e:
        print(f"[translate] Warmup failed: {e}")


def _keep_model_warm() -> None:
    if _KEEPALIVE_SECONDS <= 0:
        return

    def loop():
        while True:
            time.sleep(_KEEPALIVE_SECONDS)
            if not _GENERATE_LOCK.acquire(blocking=False):
                continue
            try:
                prompt = _build_prompt("OK", "fr-FR", "en")
                _generate_translation(prompt, 8)
                print("[translate] Keepalive complete")
            except Exception as e:
                print(f"[translate] Keepalive failed: {e}")
            finally:
                _GENERATE_LOCK.release()

    threading.Thread(target=loop, name="translate-keepalive", daemon=True).start()


_warm_model()
_keep_model_warm()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    input_chars: int | None = None
) -> str:
    """
    Translates input text using the Google TranslateGemma MLX model.
    """
    text = text.strip()
    if not text:
        return ""

    target_code = _target_code(target_lang)
    source_code = _source_code(text, source_lang)

    if _base_lang(source_code) == _base_lang(target_code):
        return text

    max_new_tokens, text_tokens, chars = _max_tokens_for(text, input_chars)

    cache_key = (source_code, target_code, text, max_new_tokens)
    cached = _cache_get(cache_key)
    if cached is not None:
        print(
            f"[translate] chars={chars} text_tokens={text_tokens} "
            f"prompt_tokens=0 max_new_tokens={max_new_tokens} generation_tokens=0 "
            f"prompt_tps=0.000 generation_tps=0.000 peak_memory=0.000 "
            f"finish_reason=cache total_ms=0 cache_hit=true"
        )
        return cached

    try:
        start = time.time()

        prompt = _build_prompt(text, source_code, target_code)
        with _GENERATE_LOCK:
            response, stats = _generate_translation(prompt, max_new_tokens)

        total_ms = int((time.time() - start) * 1000)
        print(
            f"[translate] chars={chars} text_tokens={text_tokens} "
            f"prompt_tokens={stats['prompt_tokens']} max_new_tokens={max_new_tokens} "
            f"generation_tokens={stats['generation_tokens']} "
            f"prompt_tps={stats['prompt_tps']:.3f} "
            f"generation_tps={stats['generation_tps']:.3f} "
            f"peak_memory={stats['peak_memory']:.3f} "
            f"finish_reason={stats['finish_reason']} total_ms={total_ms} "
            f"cache_hit=false"
        )

        _cache_set(cache_key, response)
        return response
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        print(error_msg)
        return f"[Error: {error_msg}]"
