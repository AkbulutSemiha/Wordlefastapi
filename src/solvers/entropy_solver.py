import math
from typing import List, Tuple
from src.solvers.base_solver import BaseSolver
from src.core.game_logic import generate_feedback

class EntropySolver(BaseSolver):
    """
    A solver that uses Information Theory (Max Entropy) to pick the next guess.
    Calculates the expected information gain for each possible word.
    """
    def predict(self, history: List[Tuple[str, Tuple[int, ...]]]) -> str:
        if history:
            last_guess, last_feedback = history[-1]
            self.filter_pool(last_guess, last_feedback)
            
        if not self.possible_words:
            return ""
        
        # Short-circuit if only one word is left
        if len(self.possible_words) == 1:
            return self.possible_words[0]

        best_guess = None
        max_entropy = float("-inf")

        # To speed up academic simulations, we could sample the pool if it's too large,
        # but for accuracy we iterate over all possible candidates.
        for guess in self.possible_words:
            entropy = self._calculate_entropy(guess)
            if entropy > max_entropy:
                max_entropy = entropy
                best_guess = guess

        return best_guess

    def _calculate_entropy(self, guess: str) -> float:
        feedback_counts = {}
        for target in self.possible_words:
            fb = generate_feedback(target, guess)
            feedback_counts[fb] = feedback_counts.get(fb, 0) + 1

        total = sum(feedback_counts.values())
        entropy = 0.0
        for count in feedback_counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy
