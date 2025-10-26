import random
import string
class RuleBasedWordleSolve:
    def __init__(self, file_path,language):
        self.read_words_from_file(path=file_path)
        self.set_letters(language)
        self.reset_possible_word()

    def set_letters(self,language):
        if language == "tr":
            self.letter_dict = {letter: idx for idx, letter in enumerate(
                ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P",
                 "R",
                 "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"])}
        elif language == "en":
            self.letter_dict = {letter: idx for idx, letter in enumerate(list(string.ascii_uppercase))}
        else:
            print(" GEÇERSİZ DİL SEÇİMİ en/tr SEÇ")

    def reset_possible_word(self):
        self.possible_words = self.words

    def read_words_from_file(self,path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                self.words = [line.strip().upper() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"Hata: {path} dosyası bulunamadı.")
            self.words = []

    @staticmethod
    def generate_feedback(target, guess):
        feedback = []
        for t, g in zip(target, guess):
            if t == g:
                feedback.append(2)  # Doğru harf, doğru pozisyon
            elif g in target:
                feedback.append(1)  # Doğru harf, yanlış pozisyon
            else:
                feedback.append(0)  # Yanlış harf
        return tuple(feedback)

    def filter_possible_words(self, guess, feedback):
        """Feedback'e göre kelime havuzunu daralt."""
        self.possible_words = [
            word
            for word in self.possible_words
            if self.generate_feedback(word, guess) == feedback
        ]

    def find_next_guess(self):
        guess = random.choice(self.possible_words)
        return guess


    def encode_data(self,data):
        encode_data = [self.letter_dict[letter] for letter in data]
        return encode_data
    def prepare_input(self,iteration_count, guess, feedback, target,previos,attempts):
        encoded_target = self.encode_data(target)
        encoded_guess =  self.encode_data(guess)
        encoded_previos = self.encode_data(previos)
        return [iteration_count, attempts] + encoded_previos + list(feedback) + encoded_guess + encoded_target


def predict_rulebased(wordleGuesses):
    solver = RuleBasedWordleSolve(file_path="Words/words_tr.txt",language="tr")
    for guess, feedback in wordleGuesses.guesses:
        solver.filter_possible_words(guess[1], tuple(feedback[1]))
        predict = solver.find_next_guess()
    print("Rule tahmin: " + predict)
    return predict