from collections import Counter
from enum import IntEnum
from typing import List, Tuple

class Feedback(IntEnum):
    GRAY = 0
    YELLOW = 1
    GREEN = 2

def generate_feedback(target: str, guess: str) -> Tuple[int, ...]:
    """
    Standard Wordle feedback logic.
    0: Gray (not in word)
    1: Yellow (in word, wrong position)
    2: Green (correct position)
    """
    if len(target) != len(guess):
        raise ValueError("Target and guess must be the same length")
    
    target = target.upper()
    guess = guess.upper()
    
    feedback = [Feedback.GRAY] * len(target)
    unmatched_target = []
    
    # First pass: Green
    for i, (t, g) in enumerate(zip(target, guess)):
        if t == g:
            feedback[i] = Feedback.GREEN
        else:
            unmatched_target.append(t)
            
    counts = Counter(unmatched_target)
    
    # Second pass: Yellow
    for i, (t, g) in enumerate(zip(target, guess)):
        if feedback[i] == Feedback.GRAY:
            if counts.get(g, 0) > 0:
                feedback[i] = Feedback.YELLOW
                counts[g] -= 1
                
    return tuple(feedback)
