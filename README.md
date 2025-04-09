# Attention-API: German to English Transformer Translation

This repository contains a **FastAPI** backend that serves a **Transformer-based German to English (de→en) translation model**.  
The model architecture is inspired by the attention mechanism and is defined in `attention_model.py`.

---

## Features

- API built with **FastAPI**
- Translation from **German to English**
- Uses a custom **Transformer model with attention**
- CORS-enabled (can be publicly accessed or restricted)
- Ready for deployment (Render, Docker, etc.)

---

## Project Structure

```
- app.py                  # FastAPI app with /translate endpoint
- attention_model.py      # Transformer model & decode_sequence function
- transformer_de_to_en_model.keras  # Trained Keras model (In Git LFS)
- source_vocab.pkl        # Source (German) vocabulary
- target_vocab.pkl        # Target (English) vocabulary
- requirements.txt        # Python dependencies
- README.md
```

---

## How It Works

- The API exposes a single POST endpoint at `/translate`
- It receives a German sentence and returns the English translation

### Example Request

```bash
POST /translate
Content-Type: application/json

{
  "text": "ich bin klug"
}
```

### Example Response

```json
{
  "translation": "i am smart"
}
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/attention-api.git
cd attention-api
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Model file is large (`.keras`, `.pkl`), make sure to use **Git LFS**:

```bash
git lfs install
git lfs pull
```

### 3. Run the FastAPI server

```bash
uvicorn app:app --reload --port 8000
```

it'll run on: `http://localhost:8000`

---

## CORS Policy

By default, the app allows all origins (`*`).  
You can restrict this in `app.py` for production environments.

---

## License

MIT License

