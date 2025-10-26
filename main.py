import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hybrid_model import predict_hybrid_model
from max_entropy import predict_max_entropy
from rulebased import predict_rulebased
import uvicorn
from typing import List


app = FastAPI()

class WordleGuess(BaseModel):
    guess: str
    feedback: List[int]

class WordleGuesses(BaseModel):
    guesses: List[WordleGuess]
@app.post("/postfeedbacklstm/")
async def postfeedbacklstm(data: WordleGuesses):
    try:
        # AI Model tahmini yapılıyor
        prediction = predict_hybrid_model(wordleGuesses=data)  # Burada 'predict' fonksiyonunuz AI modelini çalıştırır
        return {"prediction": prediction}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/postfeedbackmaxentropy/")
async def postfeedbackmaxentropy(data: WordleGuesses):
    try:
        # AI Model tahmini yapılıyor
        prediction = predict_max_entropy(wordleGuesses=data)  # Burada 'predict' fonksiyonunuz AI modelini çalıştırır
        return {"prediction": prediction}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/postfeedbackrulebased/")
async def postfeedbackrulebased(data: WordleGuesses):
    try:
        # AI Model tahmini yapılıyor
        prediction = predict_rulebased(wordleGuesses=data)  # Burada 'predict' fonksiyonunuz AI modelini çalıştırır
        return {"prediction": prediction}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

