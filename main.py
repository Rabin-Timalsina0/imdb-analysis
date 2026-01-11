from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
from tensorflow import keras

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model("trained_model.keras")

@app.get("/")
async def home(request:Request):
    return templates.TemplateResponse("index.html",{'request':request})

@app.post("/predict")
async def predict(request: Request, data: str = Form(...)):
    sequences = tokenizer.texts_to_sequences([data])
    padded_sequence = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    score = prediction[0][0]
    label = "positive" if score > 0.5 else "negative"
    return templates.TemplateResponse("index.html", {"request": request, "prediction": label})
