from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = joblib.load("mood_model.pkl")

CRISIS_KEYWORDS = [
    "bunuh diri", "pengen mati", "ingin mati", "self harm", "nyakitin diri",
    "menyakiti diri", "ga kuat lagi", "gak kuat lagi", "mengakhiri hidup"
]

def is_crisis(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def reply_for(mood: str, msg: str) -> dict:
    mood = (mood or "neutral").lower()
    msg = (msg or "").strip()

    # Guardrail krisis
    if is_crisis(msg):
        return {
            "reply": (
                "Aku ikut prihatin kamu lagi ngerasa berat. Aku bukan pengganti bantuan profesional, "
                "tapi keselamatan kamu yang utama. Kalau kamu merasa berisiko menyakiti diri, "
                "tolong segera hubungi orang terdekat sekarang atau layanan darurat setempat."
            )
        }

    # Respon berbasis mood
    if mood == "happy":
        return {
            "reply": (
                "Senang dengernya 😊 Mau ceritain, hal apa yang paling bikin kamu merasa baik hari ini?"
            )
        }

    if mood == "bad":
        return {
            "reply": (
                "Aku denger kamu lagi berat. Kita ambil langkah kecil dulu ya. "
                "Dari 1–5, seberapa berat rasanya sekarang? (1 ringan, 5 berat banget)"
            )
        }

    # neutral / default
    return {
        "reply": (
            "Oke, kita pelan-pelan ya 🙂 Yang paling kerasa sekarang lebih ke capek fisik, stres pikiran, "
            "atau kepikiran sesuatu?"
        )
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(payload: dict):
    X = pd.DataFrame([{
        "tidur_jam": payload["tidur_jam"],
        "stres": payload["stres"],
        "capek": payload["capek"],
        "aktivitas": payload["aktivitas"]
    }])
    pred = model.predict(X)[0]
    return {"mood": pred}

@app.post("/chat")
def chat(payload: dict):
    mood = payload.get("mood", "neutral")
    message = payload.get("message", "")
    return reply_for(mood, message)
