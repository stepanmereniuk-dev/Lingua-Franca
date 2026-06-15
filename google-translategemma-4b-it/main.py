import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "google/translategemma-4b-it"


def load_model():
    """Load the model on GPU if available, otherwise on CPU."""
    if torch.cuda.is_available():
        print("Loading on GPU (4-bit, fits in 6 GB VRAM)...")
        return Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            # Shrink the model from 8 GB to ~3 GB so it fits on the card
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
    else:
        print("No GPU found, loading on CPU (slow)...")
        return Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="cpu")


model = load_model()
processor = AutoProcessor.from_pretrained(MODEL_ID)


def translate(text, source_lang="en", target_lang="es"):
    """Translate text between two languages (codes like en, es, fr, de, ja)."""
    # 1. Build the message in the format this model expects
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        }],
    }]

    # 2. Turn the message into numbers (tokens) and move them to the GPU
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 3. Let the model generate the translation, one token at a time
    output = model.generate(**inputs, max_new_tokens=256)

    # 4. Turn the new tokens back into text (skip the prompt at the start)
    prompt_length = inputs["input_ids"].shape[-1]
    return processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(translate("Hello, how are you today?", "en", "es"))
    print(translate("The weather is beautiful this morning.", "en", "fr"))
