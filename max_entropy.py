import math
import string
from collections import Counter


class MaxEnropyWordleSolve:
    def __init__(self, file_path,language):
        self.file_path = file_path
        self.read_words_from_file()
        self.language=language
        self.set_letters()
        self.reset_possible_word()

    def set_letters(self):
        if self.language == "tr":
            self.letter_dict = {letter: idx for idx, letter in enumerate(
                ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P",
                 "R",
                 "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"])}
        elif self.language == "en":
            self.letter_dict = {letter: idx for idx, letter in enumerate(list(string.ascii_uppercase))}
        else:
            print(" GEÇERSİZ DİL SEÇİMİ en/tr SEÇ")

    def reset_possible_word(self):
        self.possible_words = self.words

    def read_words_from_file(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.words = [line.strip().upper() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"Hata: {self.file_path} dosyası bulunamadı.")
            self.words = []

    @staticmethod
    def generate_feedback(target, guess):
        feedback = [0] * len(target)
        unmatched = []
        for i, (t, g) in enumerate(zip(target, guess)):
            if t == g:
                feedback[i] = 2  # green
            else:
                unmatched.append(t)
        counts = Counter(unmatched)

        for i, (t, g) in enumerate(zip(target, guess)):
            if feedback[i] == 0:  # not green
                if counts.get(g, 0) > 0:
                    feedback[i] = 1  # yellow
                    counts[g] -= 1
                else:
                    feedback[i] = 0  # gray

        return tuple(feedback)

    def filter_possible_words(self, guess, feedback):
        """Feedback'e göre kelime havuzunu daralt."""
        self.possible_words = [
            word
            for word in self.possible_words
            if self.generate_feedback(word, guess) == feedback
        ]

    def calculate_entropy(self, guess, possible_words):
        feedback_counts = {}
        for target in possible_words:
            feedback = self.generate_feedback(target, guess)
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1

        total = sum(feedback_counts.values())
        entropy = 0
        for count in feedback_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    def find_max_entropy_guess(self):
        """Paralel işlem olmadan maksimum entropi hesaplar."""
        best_guess = None
        max_entropy = float("-inf")

        for guess in self.possible_words:
            entropy = self.calculate_entropy(guess, self.possible_words)

            if entropy > max_entropy:
                max_entropy = entropy
                best_guess = guess

        return best_guess, max_entropy
    def encode_data(self,data):
        encode_data = [self.letter_dict[letter] for letter in data]
        return encode_data
    def prepare_input(self,iteration_count, guess, feedback, target,previos,attempts):
        # Harfleri sayısal değerlere çevir
        encoded_target = self.encode_data(target)
        encoded_guess = self.encode_data(guess)
        encoded_previos = self.encode_data(previos)
        return [iteration_count, attempts] + encoded_previos + list(feedback) + encoded_guess + encoded_target


