import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from wordle_predict import predict
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
        prediction = predict(wordleGuesses=data)  # Burada 'predict' fonksiyonunuz AI modelini çalıştırır
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
# Uygulamayı çalıştırma
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


"""import json
import socket
from wordle_predict import predict
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 5000))
    server.listen(1)
    print("Server başladı, bağlantı bekleniyor...")

    conn, addr = server.accept()
    print(f"Bağlantı alındı: {addr}")

    while True:
        data = conn.recv(1024).decode("utf-8")
        if not data:
            break
        received_data = json.loads(data)  # JSON string → Python list
        print(f"Unity'den Gelen Tahmin: {received_data}")

        # AI Model Tahmini
        prediction = predict(data=received_data)

        serialized_data = json.dumps(prediction, ensure_ascii=False).encode("utf-8")  # ensure_ascii=False ekledik
        conn.send(serialized_data)

    conn.close()

if __name__ == "__main__":
    start_server()"""