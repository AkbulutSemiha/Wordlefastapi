from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseSolver(ABC):
    def __init__(self, words: List[str]):
        self.all_words = words
        self.possible_words = list(words)

    def reset(self):
        """Resets the possible word pool for a new game."""
        self.possible_words = list(self.all_words)

    @abstractmethod
    def predict(self, history: List[Tuple[str, Tuple[int, ...]]]) -> str:
        """
        Takes the history of guesses and feedbacks, filters the word pool,
        and returns the next predicted guess.
        history format: [("GUESS", (0, 1, 2, ...)), ...]
        """
        pass

    def filter_pool(self, guess: str, feedback: Tuple[int, ...]):
        """Standard filtering logic shared by most solvers."""
        from src.core.game_logic import generate_feedback
        self.possible_words = [
            word for word in self.possible_words
            if generate_feedback(word, guess) == feedback
        ]
