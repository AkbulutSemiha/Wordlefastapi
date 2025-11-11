import csv
from collections import Counter
from pydantic import BaseModel
from hybrid_model import OurHybridModel
from max_entropy import MaxEnropyWordleSolve
from rulebased import RuleBasedWordleSolve
from typing import List
import pandas as pd
import random
class WordleGuess(BaseModel):
    guess: str
    feedback: List[int]

class WordleGuesses(BaseModel):
    guesses: List[WordleGuess]


def read_words_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            words = [line.strip().upper() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
        words = []
    finally:
        file.close()
    return words
def read_frequencies_words_from_file(words):
    df = pd.read_csv("Words/five_letter_word_frequencies.csv")

    alpha = 0.5  # frequency öncelikli
    # Frequency’yi yüzdelik olarak normalize et
    df["freq_norm"] = df["frequency"].rank(pct=True)
    df["score"] = df["freq_norm"] * alpha + df["dispersion"] * (1 - alpha)
    answer_pool = df.sort_values("score", ascending=False)
    tr_map = str.maketrans("çğiöşü", "ÇĞİÖŞÜ")
    answer_pool["word"] = answer_pool["word"].apply(lambda x: x.translate(tr_map).upper())

    filtered = answer_pool[answer_pool["word"].isin(words)]
    filtered = filtered.head(1000)
    return filtered["word"].tolist()

def submit_feedback(current_guess, target_word):
    feedback = [0] * len(target_word)
    unmatched = []
    for i, (t, g) in enumerate(zip(target_word,current_guess)):
        if t == g:
            feedback[i] = 2   # Doğru harf, doğru pozisyon
        else:
            unmatched.append(t)
    counts = Counter(unmatched)
    for i, (t, g) in enumerate(zip(target_word, current_guess)):
        if feedback[i] == 0:  # not green
            if counts.get(g, 0) > 0:
                feedback[i] = 1  # yellow
                counts[g] -= 1
            else:
                feedback[i] = 0  # gray

    return tuple(feedback)
def write_to_csv(file_name, data):
    with open(file_name, mode='w', newline='',encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Target Word", "AI Prediction","Hybrid Prediction", "Rule-Based Prediction", "Entropy Prediction", "Steps"])
        for row in data:
            writer.writerow(row)

def predict_max_entropy(solver,wordleGuesses):
    for guess, feedback in wordleGuesses.guesses:
        solver.filter_possible_words(guess[1], tuple(feedback[1]))
        predict, _ = solver.find_max_entropy_guess()
    print("Entropy tahmin: " +predict)
    return predict

def predict_hybrid_model(solver, wordleGuesses):  # input[[]] liste içinde liste şeklinde
    solver.filter_possible_words(wordleGuesses.guesses)
    ai,final_prediction= solver.find_next_guess(wordleGuesses=wordleGuesses)
    print("Hibrik tahmin: "+final_prediction)
    return ai,final_prediction

def predict_rulebased(solver, wordleGuesses):
    for guess, feedback in wordleGuesses.guesses:
        solver.filter_possible_words(guess[1], tuple(feedback[1]))
        predict = solver.find_next_guess()
    print("Rule tahmin: " + predict)
    return predict


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)

    all_words = read_words_from_file(file_path="Words/words_tr.txt")
    freq_words = read_frequencies_words_from_file(all_words)
    hybrid_solver = OurHybridModel(file_path="Words/words_tr.txt", language="tr")
    rule_solver = RuleBasedWordleSolve(file_path="Words/words_tr.txt", language="tr")
    entropy_solver = MaxEnropyWordleSolve(file_path="Words/words_tr.txt", language="tr")

    max_steps = 6
    simulations = []

    for i in range(1000):  # 1000 farklı simülasyon
        print("Simülasyon step: "+ str(i))
        target_word = freq_words[i]
        print(target_word)
        ai_g,rl_g,ent_g = [],[],[]
        first_guess= random.choice(all_words)
        print(first_guess)
        ai_guess =first_guess  # İlk AI tahmini
        hbrid_guess=first_guess
        rule_guess = first_guess  # İlk kural tabanlı tahmin
        entropy_guess =first_guess # İlk entropi tabanlı tahmin
        iteration_count = 1

        # Her yeni hedef kelime için solver’ların durumunu sıfırla
        hybrid_solver.reset_possible_word()
        rule_solver.reset_possible_word()
        entropy_solver.reset_possible_word()
        simulations.append([target_word, ai_guess, hbrid_guess, rule_guess, entropy_guess, iteration_count])
        while iteration_count < max_steps:
            ai_feedback = submit_feedback(hbrid_guess, target_word)
            ai_g.append(WordleGuess(guess=hbrid_guess, feedback=ai_feedback))
            ai_guess,hbrid_guess = predict_hybrid_model(solver=hybrid_solver,wordleGuesses=WordleGuesses(guesses=ai_g))  # AI bir sonraki tahmini

            rule_feedback = submit_feedback(rule_guess, target_word)
            rl_g.append(WordleGuess(guess=rule_guess, feedback=rule_feedback))
            rule_guess = predict_rulebased(solver=rule_solver,wordleGuesses=WordleGuesses(guesses=rl_g))  # Kural tabanlı bir sonraki tahmini

            entropy_feedback = submit_feedback(entropy_guess, target_word)
            ent_g.append(WordleGuess(guess=entropy_guess, feedback=entropy_feedback))
            entropy_guess = predict_max_entropy(solver=entropy_solver,wordleGuesses=WordleGuesses(guesses=ent_g))  # Entropi tabanlı bir sonraki tahmini

            simulations.append([target_word, ai_guess,hbrid_guess, rule_guess, entropy_guess, iteration_count])
            iteration_count +=1
    write_to_csv("Cosine1000.csv", simulations)



