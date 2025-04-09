from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from attention_model import decode_sequence

app = FastAPI()

# Allow CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(req: TranslateRequest):
    translation = decode_sequence(req.text)
    return {"translation": translation}
