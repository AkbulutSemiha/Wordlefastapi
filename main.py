from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn

from src.infra.word_repository import WordRepository
from src.solvers.rule_based_solver import RuleBasedSolver
from src.solvers.entropy_solver import EntropySolver
from src.solvers.hybrid_lstm_solver import HybridLSTMSolver

app = FastAPI(title="Wordle Academic Solver API")

# Global dependencies (Singletons for efficiency)
repo = WordRepository(data_dir="Words")
ALL_WORDS = repo.get_words(language="tr")
MODEL_PATH = "src/models/tr_LSTMmodel_100epoch.pth"

class WordleGuess(BaseModel):
    guess: str
    feedback: List[int]

class WordleGuesses(BaseModel):
    guesses: List[WordleGuess]

def prepare_history(data: WordleGuesses) -> List[Tuple[str, Tuple[int, ...]]]:
    return [(g.guess, tuple(g.feedback)) for g in data.guesses]

@app.post("/postfeedbacklstm/")
async def postfeedbacklstm(data: WordleGuesses):
    try:
        solver = HybridLSTMSolver(words=ALL_WORDS, model_path=MODEL_PATH)
        history = prepare_history(data)
        prediction = solver.predict(history)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid Prediction error: {str(e)}")

@app.post("/postfeedbackmaxentropy/")
async def postfeedbackmaxentropy(data: WordleGuesses):
    try:
        solver = EntropySolver(words=ALL_WORDS)
        history = prepare_history(data)
        prediction = solver.predict(history)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entropy Prediction error: {str(e)}")

@app.post("/postfeedbackrulebased/")
async def postfeedbackrulebased(data: WordleGuesses):
    try:
        solver = RuleBasedSolver(words=ALL_WORDS)
        history = prepare_history(data)
        prediction = solver.predict(history)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rule-based Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
