import csv

from pydantic import BaseModel
from wordle_predict import predict
from max_entropy import predict_max_entropy
from rulebased import predict_rulebased
from typing import List
import json
import requests
import random
class WordleGuess(BaseModel):
    guess: str
    feedback: List[int]

class WordleGuesses(BaseModel):
    guesses: List[WordleGuess]


def read_words_from_file():
    file_path = "dataset/default_words.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            words = [line.strip().upper() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
        words = []
    finally:
        file.close()
    return words

def submit_feedback(current_guess, target_word):
    feedback = []
    for t, g in zip(target_word,current_guess):
        if t == g:
            feedback.append(2)  # Doğru harf, doğru pozisyon
        elif g in target_word:
            feedback.append(1)  # Doğru harf, yanlış pozisyon
        else:
            feedback.append(0)  # Yanlış harf

    return feedback
def write_to_csv(file_name, data):
    with open(file_name, mode='w', newline='',encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Target Word", "AI Prediction", "Rule-Based Prediction", "Entropy Prediction", "Steps"])
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    words = read_words_from_file()
    max_steps = 7
    simulations = []

    for _ in range(1000):  # 1000 farklı simülasyon
        print("Simülasyon step: "+ str(_))
        target_word = random.choice(words)
        ai_g,rl_g,ent_g = [],[],[]
        first_guess= random.choice(words)
        ai_guess =first_guess  # İlk AI tahmini
        rule_guess = first_guess  # İlk kural tabanlı tahmin
        entropy_guess =first_guess # İlk entropi tabanlı tahmin
        iteration_count = 1

        while iteration_count < max_steps:
            ai_feedback = submit_feedback(ai_guess, target_word)
            rule_feedback = submit_feedback(rule_guess, target_word)
            entropy_feedback = submit_feedback(entropy_guess, target_word)

            ai_g.append(WordleGuess(guess=ai_guess, feedback=ai_feedback))
            rl_g.append(WordleGuess(guess=rule_guess, feedback=rule_feedback))
            ent_g.append(WordleGuess(guess=entropy_guess, feedback=entropy_feedback))

            ai_guess = predict(wordleGuesses=WordleGuesses(guesses=ai_g))  # AI bir sonraki tahmini
            rule_guess = predict_rulebased(wordleGuesses=WordleGuesses(guesses=rl_g))  # Kural tabanlı bir sonraki tahmini
            entropy_guess = predict_max_entropy(wordleGuesses=WordleGuesses(guesses=ent_g))  # Entropi tabanlı bir sonraki tahmini
            simulations.append([target_word, ai_guess, rule_guess, entropy_guess, iteration_count])
            iteration_count +=1


    write_to_csv("16_Ocakwordle_simulations.csv", simulations)



