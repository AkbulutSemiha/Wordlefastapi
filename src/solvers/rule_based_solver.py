import random
from typing import List, Tuple
from src.solvers.base_solver import BaseSolver

class RuleBasedSolver(BaseSolver):
    """
    A simple rule-based solver that picks a random word from the remaining possible pool.
    Used as a baseline for academic comparison.
    """
    def predict(self, history: List[Tuple[str, Tuple[int, ...]]]) -> str:
        # If history is provided, filter the pool based on the last guess
        if history:
            last_guess, last_feedback = history[-1]
            self.filter_pool(last_guess, last_feedback)
            
        if not self.possible_words:
            return ""
            
        return random.choice(self.possible_words)
