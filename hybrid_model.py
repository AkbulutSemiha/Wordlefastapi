import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
class WordleLSTM(nn.Module):
    """
    LSTM-based neural network for predicting Wordle words.

    Architecture rationale:
    - Embedding layer: Converts integer-encoded letters into dense vectors.
      This allows the model to learn semantic similarities between letters or feedback.
    - LSTM: Captures sequential dependencies between consecutive guesses within a game.
      Since Wordle guesses follow a sequential pattern (later guesses depend on earlier feedback),
      LSTM is suitable for learning these temporal relationships.
    - Dropout layers: Reduce overfitting by randomly dropping units during training.
    - Fully connected layers: Transform LSTM hidden states into predictions for each letter.
      The output dimension is 5 letters * 29 possible classes (vocab_size).
    - Packed sequences: Handle variable-length games efficiently.
    """
    def __init__(
        self,
        vocab_size=29,
        embedding_dim=16,
        input_dim=10,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 5 * vocab_size  # 5 Letter  * 29 class = 145

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm_input_dim = input_dim * embedding_dim

        self.lstm = nn.LSTM(
            self.lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.post_lstm_dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x, lengths):

        batch_size, seq_len, fifteen_dim = x.size()

        x_embed = self.embedding(x)
        x_embed = x_embed.view(batch_size, seq_len, -1)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_embed,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed_x)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.post_lstm_dropout(out)

        out = F.relu(self.fc1(out))
        out = self.post_lstm_dropout(out)
        out = self.fc2(out)

        b_size, seq_len, _ = out.size()
        out = out.view(b_size, seq_len, 5, -1)

        return out



class OurHybridModel:
    def __init__(self, file_path, language):
        self.read_words_from_file(path=file_path)
        self.set_parameters_according_language(language=language)
        self.reset_possible_word()


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
    def generate_feedback(self,target, guess):
        feedback = []
        for t, g in zip(target, guess):
            if t == g:
                feedback.append(2)  # Doğru harf, doğru pozisyon
            elif g in target:
                feedback.append(1)  # Doğru harf, yanlış pozisyon
            else:
                feedback.append(0)  # Yanlış harf
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

        def word_distance(word1, word2):
            vec1 = np.array(word1)
            vec2 = np.array(word2)
            if len(vec1) == 0 or len(vec2) == 0:
                return 0  # Eğer kelime boşsa benzerlik sıfırdır.
            euclidean_distance = np.linalg.norm(vec1 - vec2)
            return euclidean_distance

        min_similarity = 100000  # Başlangıçta benzerlik değeri en düşük
        most_similar_word = None

        for word in remaining_words:
            similarity = word_distance(self.encode_word(prediction), self.encode_word(word))
            if similarity < min_similarity:
                min_similarity = similarity
                most_similar_word = word

        return most_similar_word


    def find_next_guess(self,wordleGuesses):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WordleLSTM(
            vocab_size=len(self.letter2index),  # Harf sayısı
            embedding_dim=16,  # Embedding boyutu
            input_dim=10,  # Girdi boyutu
            hidden_dim=256,  # LSTM gizli boyutu
            num_layers=4,  # LSTM katman sayısı
            dropout=0.3
        ).to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()
        model_input = self.prepare_input_for_model(wordleGuesses=wordleGuesses)
        sequence = torch.tensor(model_input, dtype=torch.long)
        sequence = sequence.unsqueeze(0)
        lengths = torch.tensor([len(model_input)], dtype=torch.long)
        sequence = sequence.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            logits = model(sequence, lengths)  # (1, 3, 5, 29)
            predictions = torch.argmax(logits, dim=-1)  # (1, 3, 5)
            decoded_predictions = []
            for t in range(predictions.shape[1]):  # seq_len = 3
                step_preds = predictions[0, t]
                decoded_predictions.append(self.decode_model_output(step_preds.tolist()))
        prediction = self.get_word_distance(decoded_predictions[-1],self.possible_words)
        return prediction
def predict_hybrid_model(wordleGuesses):  # input[[]] liste içinde liste şeklinde
    solver = OurHybridModel(file_path="Words/words_tr.txt",language="tr")
    solver.filter_possible_words(wordleGuesses.guesses)
    final_prediction= solver.find_next_guess(wordleGuesses=wordleGuesses)
    print("Hibrik tahmin: "+final_prediction)
    return final_prediction




