import random
import string
from collections import Counter


class HumanBehaviour:
    def __init__(self, file_path,lng):
        self.file_path = file_path
        self.language=lng
        self.read_words_from_file()
        self.set_letters()

        
    def reset_possible_word(self):
        self.possible_words = self.words
    def set_letters(self):
        if self.language == "tr":
            self.letter_dict = {letter: idx for idx, letter in enumerate(
                ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P", "R",
                "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"])} 
        elif self.language =="en":
            self.letter_dict = {letter: idx for idx, letter in enumerate(list(string.ascii_uppercase))}
        else:
            print(" GEÇERSİZ DİL SEÇİMİ en/tr SEÇ")
                                                        
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

    def find_next_guess(self):
        guess = random.choice(self.possible_words)
        return guess

    def encode_data(self, data):
        encode_data = [self.letter_dict[letter] for letter in data]
        return encode_data

    def prepare_input(self, iteration_count, guess, feedback, target, previos, attempts):
        # Harfleri sayısal değerlere çevir
        encoded_target = self.encode_data(target)  # [self.letter_dict[letter] for letter in target]
        encoded_guess = self.encode_data(guess)  # [self.letter_dict[letter] for letter in guess]
        encoded_previos = self.encode_data(previos)
        return [iteration_count, attempts] + encoded_previos + list(feedback) + encoded_guess + encoded_target


    def play_wordle_like_human(self,target_word,  guess, iteration_count):
        attempts = 0
        print("Wordle çözüm başlıyor...\n")
        self.reset_possible_word()
        dataset_list = []
        while True:
            previous = guess
            feedback = self.generate_feedback(target_word, previous)
            self.filter_possible_words(previous, feedback)
            guess = self.find_next_guess()

            print(f"gameid: {iteration_count} attempt_index:{attempts} Previous:{previous} feedback:{feedback} guess: {guess} Target: {target_word} ")
            # Hedef kelime bulunduysa dur
            if guess == target_word or attempts == 6:
                dataset_list.append(
                    self.prepare_input(iteration_count=iteration_count, previos=previous, feedback=feedback, guess=guess,
                                          target=target_word, attempts=attempts))
                print(f"\nHedef kelime '{target_word}' {attempts} tahminde bulundu!")
                break
            # Feedback al ve havuzu daralt
            dataset_list.append(
                self.prepare_input(iteration_count=iteration_count, previos=previous, feedback=feedback, guess=guess,
                                      target=target_word, attempts=attempts))
            attempts += 1
        return dataset_list



