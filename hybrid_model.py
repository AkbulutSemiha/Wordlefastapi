import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
class WordleLSTM(nn.Module):
    """
    LSTM-based neural network for predicting Wordle words.

    Architecture rationale:

    - Separate Embedding layers:
        * Letter embedding converts integer-encoded letters into dense vectors.
          This allows the model to learn semantic relationships between letters.
        * Feedback embedding converts integer feedback values (0=wrong, 1=wrong position, 2=correct)
          into dense vectors, allowing the model to treat feedback as a separate concept.

    - LSTM:
        * Captures sequential dependencies between consecutive guesses within a Wordle game.
        * Later guesses depend on earlier feedback, which the LSTM hidden states can learn.
        * Packed sequences are used to efficiently handle variable-length games.

    - Dropout layers:
        * Applied after LSTM and fully connected layers to reduce overfitting
          and improve generalization.

    - Fully connected layers:
        * Transform LSTM hidden states into predictions for each of the 5 letter positions.
        * Output dimension = 5 letters * vocab_size (number of possible letters)
        * Softmax applied during training to compute probability distribution over letters.

    - Overall:
        * The model can predict a 5-letter word at each timestep based on prior guesses and feedback.
        * This design separates letter and feedback representations while capturing temporal dependencies.

    """
    def __init__(
        self,
        vocab_size=29,           # Harf sayısı
        letter_embedding_dim=16, # Harf embedding boyutu
        feedback_embedding_dim=4,# Feedback embedding boyutu (0,1,2)
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Çıkış boyutu: 5 harf * vocab_size
        self.output_dim = 5 * vocab_size

        # Harf ve feedback embedding
        self.letter_embedding = nn.Embedding(vocab_size, letter_embedding_dim)
        self.feedback_embedding = nn.Embedding(3, feedback_embedding_dim)

        # LSTM input boyutu = 5 harf * letter_emb + 5 feedback * feedback_emb
        self.lstm_input_dim = 5 * letter_embedding_dim + 5 * feedback_embedding_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.post_lstm_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x, lengths):
        """
        x: (batch_size, seq_len, 10) -> 5 harf + 5 feedback
        lengths: (batch_size,) sequence uzunlukları
        """
        batch_size, seq_len, _ = x.size()

        # Harf ve feedback ayrımı
        letters = x[:, :, :5]   # pg1..pg5
        feedback = x[:, :, 5:]  # l1..l5

        # Embedding
        letters_emb = self.letter_embedding(letters)       # (B, S, 5, letter_embedding_dim)
        feedback_emb = self.feedback_embedding(feedback)   # (B, S, 5, feedback_embedding_dim)

        # Concatenate embeddingler
        x_embed = torch.cat([letters_emb, feedback_emb], dim=-1)  # (B, S, 5, letter+fb emb)
        x_embed = x_embed.view(batch_size, seq_len, -1)            # (B, S, LSTM_input_dim)

        # Packed sequence ile LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_embed,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, (h, c) = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Dropout + FC
        out = self.post_lstm_dropout(out)
        out = F.relu(self.fc1(out))
        out = self.post_lstm_dropout(out)
        out = self.fc2(out)

        # Çıkış reshape: (B, S, 5 harf, vocab_size)
        out = out.view(batch_size, seq_len, 5, -1)

        return out


class OurHybridModel:
    def __init__(self, file_path, language):
        self.read_words_from_file(path=file_path)
        self.set_parameters_according_language(language=language)
        self.reset_possible_word()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model_once()
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def _load_model_once(self):
        """Load model safely and only once."""
        if not os.path.exists(self.model_path):
            print(f"Uyarı: Model dosyası bulunamadı: {self.model_path}")
            return None

        model = WordleLSTM(
            vocab_size=29,  # letter numbers in alphabet
            letter_embedding_dim=16,  # Previous Guess Embedding dimension
            feedback_embedding_dim=4,  # Feedback Embedding dimension
            hidden_dim=256,  # LSTM hidden dimension
            num_layers=4,  # LSTM layer number
            dropout=0.3  # Drop out
        ).to(self.device)

        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            print(f"Model yüklendi: {self.model_path}")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            model = None
        return model
    def read_words_from_file(self,path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                self.words = [line.strip().upper() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"Hata: {path} dosyası bulunamadı.")
            self.words = []
    def set_parameters_according_language(self,language):
        if language == "tr":
            self.letter2index = {letter: idx for idx, letter in enumerate(
                ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P",
                 "R",
                 "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"])}
            self.index2letter = {i: ch for ch, i in self.letter2index.items()}
            self.model_path= "model/tr_LSTMmodel_100epoch.pth"
        elif language == "en":
            self.letter2index = {letter: idx for idx, letter in enumerate(list(string.ascii_uppercase))}
            self.index2letter = {i: ch for ch, i in self.letter2index.items()}
            self.model_path= "model/en_LSTMmodel_100epoch.pth"

        else:
            print(" GEÇERSİZ DİL SEÇİMİ en/tr SEÇ")
    def reset_possible_word(self):
        self.possible_words = self.words

    def encode_word(self,word):
        encode = [self.letter2index[ch] for ch in word]
        return encode

    def decode_model_output(self,encoded_word):
        decode = "".join(self.index2letter[idx] for idx in encoded_word)
        return decode

    def prepare_input_for_model(self,wordleGuesses):
        all_input = []
        for wordleGuess in wordleGuesses.guesses:
            encoded = self.encode_word(wordleGuess.guess)
            item = encoded + wordleGuess.feedback
            all_input.append(item)
        return all_input

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

    def filter_possible_words(self,guesses):
        """Feedback'e göre kelime havuzunu daralt."""
        for guess, feedback in guesses:
            self.possible_words = [
                word
                for word in self.possible_words
                if self.generate_feedback(word, guess[1]) == tuple(feedback[1])
            ]

    def get_word_distance(self,prediction, remaining_words):

        """
        lstm_pred_indices: LSTM'nin tahmin ettiği 5 harfli kelime indeksleri [i1,i2,i3,i4,i5]
        word_pool: filtrelenmiş geçerli kelimeler listesi
        """
        num_letters = 29  # Türkçedeki harf sayısı (a-z + ç, ğ, ş, ö, ü, ı)
        embed_dim = 10  # Her harf için 10 boyutlu vektör

        self.embedding = nn.Embedding(num_letters, embed_dim)
        self.embedding.eval()  # deterministik embedding
        with torch.no_grad():
            # LSTM tahmini kelimenin embedding vektörleri
            pred_embeds = [self.embedding(torch.tensor(i).to(self.device)).cpu().numpy() for i in prediction]
            pred_vec = np.concatenate(pred_embeds)  # veya np.mean(pred_embeds, axis=0) yapabilirsin

            min_distance = float("inf")
            closest_word = None

            for word in remaining_words:
                word_indices = [self.letter2index[ch] for ch in word]
                word_embeds = [self.embedding(torch.tensor(i).to(self.device)).cpu().numpy() for i in word_indices]
                word_vec = np.concatenate(word_embeds)  # aynı şekilde concat veya mean

                distance = np.linalg.norm(pred_vec - word_vec)  # Euclidean distance
                if distance < min_distance:
                    min_distance = distance
                    closest_word = word

        return closest_word


    def find_closest_word_cosine(self, lstm_pred_indices, word_pool):

        num_letters = 29  # Türkçedeki harf sayısı (a-z + ç, ğ, ş, ö, ü, ı)
        embed_dim = 10  # Her harf için 10 boyutlu vektör

        self.embedding = nn.Embedding(num_letters, embed_dim)
        self.embedding.eval()
        """
        lstm_pred_indices: LSTM çıktısı indeks listesi [i1, i2, i3, i4, i5]
        word_pool: filtrelenmiş geçerli kelimeler listesi
        """
        # LSTM tahmini harfleri embedding vektörüne dönüştür
        pred_embeds = [self.embedding(torch.tensor(i)).detach().cpu().numpy() for i in lstm_pred_indices]
        pred_vec = np.mean(pred_embeds, axis=0)  # pozisyonları birleştir (ortalama)

        best_word = None
        best_sim = -np.inf

        for word in word_pool:
            word_indices = [self.letter2index[ch] for ch in word]
            word_embeds = [self.embedding(torch.tensor(i)).detach().cpu().numpy() for i in word_indices]
            word_vec = np.mean(word_embeds, axis=0)

            # Cosine similarity
            sim = np.dot(pred_vec, word_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(word_vec))

            if sim > best_sim:
                best_sim = sim
                best_word = word

        return best_word

    def find_next_guess(self, wordleGuesses):
        model_input = self.prepare_input_for_model(wordleGuesses)
        sequence = torch.tensor(model_input, dtype=torch.long).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(model_input)], dtype=torch.long).to(self.device)

        self.model.eval()  # deterministik inference
        with torch.no_grad():
            logits = self.model(sequence, lengths)  # (1, seq_len, 5, vocab_size)
            predictions = torch.argmax(logits, dim=-1)  # (1, seq_len, 5)
            step_preds = predictions[0, -1].tolist()  # sadece son adım
            model_predict = self.decode_model_output(step_preds)
            closest_word = self.find_closest_word_cosine(step_preds, self.possible_words)
            #closest_word = self.get_word_distance(step_preds,self.possible_words)
        print("LSTM tahmini:", model_predict)
        print("En yakın kelime:", closest_word)
        return model_predict, closest_word
