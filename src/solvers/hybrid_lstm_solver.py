import os
import torch
import numpy as np
import random
from typing import List, Tuple
from src.solvers.base_solver import BaseSolver
from src.models.wordle_lstm import WordleLSTM # Importing the network structure

class HybridLSTMSolver(BaseSolver):
    """
    A hybrid solver that uses an LSTM to predict the target word,
    and then finds the closest valid word in the possible pool using embeddings.
    """
    def __init__(self, words: List[str], model_path: str, device: str = None):
        super().__init__(words)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = model_path
        self.model = self._load_model()
        self.letter2index = self._get_tr_mapping() # Default to TR as per project context
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Hata: Model bulunamadı {self.model_path}")
            return None
            
        model = WordleLSTM(
            vocab_size=29,
            letter_embedding_dim=16,
            feedback_embedding_dim=4,
            hidden_dim=256,
            num_layers=4,
            dropout=0.3
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model

    def _get_tr_mapping(self):
        letters = ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"]
        return {l: i for i, l in enumerate(letters)}

    def predict(self, history: List[Tuple[str, Tuple[int, ...]]]) -> str:
        if history:
            last_guess, last_feedback = history[-1]
            self.filter_pool(last_guess, last_feedback)
            
        if not self.possible_words:
            return ""
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]

        # Prepare LSTM input from history
        lstm_input = self._prepare_lstm_input(history)
        if not lstm_input:
            # Fallback for first guess if no history
            return random.choice(self.possible_words)

        sequence = torch.tensor([lstm_input], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(lstm_input)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(sequence, lengths)
            predictions = torch.argmax(logits, dim=-1) # (1, seq_len, 5)
            last_step_preds = predictions[0, -1].tolist()

            # Fix: Use the corrected distance logic (now using model embeddings)
            return self._get_closest_word(last_step_preds)

    def _prepare_lstm_input(self, history: List[Tuple[str, Tuple[int, ...]]]):
        encoded_history = []
        for guess, feedback in history:
            encoded_guess = [self.letter2index[c] for c in guess]
            item = encoded_guess + list(feedback)
            encoded_history.append(item)
        return encoded_history

    def _get_closest_word(self, prediction_indices: List[int]) -> str:
        self.model.eval()
        with torch.no_grad():
            pred_indices_tensor = torch.tensor(prediction_indices).to(self.device)
            pred_embeds = self.model.letter_embedding(pred_indices_tensor)
            pred_vec = pred_embeds.view(-1).cpu().numpy()

            min_dist = float("inf")
            best_word = self.possible_words[0]

            for word in self.possible_words:
                word_indices = [self.letter2index[c] for c in word]
                word_tensor = torch.tensor(word_indices).to(self.device)
                word_embeds = self.model.letter_embedding(word_tensor)
                word_vec = word_embeds.view(-1).cpu().numpy()
                
                dist = np.linalg.norm(pred_vec - word_vec)
                if dist < min_dist:
                    min_dist = dist
                    best_word = word
        return best_word
