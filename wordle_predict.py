import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

class WordleLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=29,
        embedding_dim=16,
        input_dim=10,#15
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        """
        vocab_size : Her bir harf vb. kategorik değerin toplam sayısı (29)
        embedding_dim : Her bir harf indeksini gömdüğümüz vektör boyutu
        input_dim : Her adımda modele giren özelliğin sayısı (15)
        hidden_dim : LSTM katmanlarındaki gizli boyut
        num_layers : LSTM katman sayısı
        dropout : LSTM katmanları arasındaki dropout oranı
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = 5 * vocab_size  # 5 harf * 29 sınıf = 145

        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Çok katmanlı LSTM
        # Gömülen 15 özelliği (her biri embedding_dim boyutlu) birleştireceğiz
        # bu nedenle LSTM'e girecek boyut = 15 * embedding_dim
        self.lstm_input_dim = input_dim * embedding_dim

        self.lstm = nn.LSTM(
            self.lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # LSTM sonrası ek dropout katmanı
        self.post_lstm_dropout = nn.Dropout(p=dropout)

        # İki aşamalı FF katmanı
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x, lengths):
        """
        x: (batch_size, seq_len, 15) -- her değer [0..28] arası indeks
        lengths: her batch içindeki gerçek sekans uzunlukları
        """
        batch_size, seq_len, fifteen_dim = x.size()

        # Embedding katmanı: (b, seq_len, 15) -> (b, seq_len, 15, embedding_dim)
        x_embed = self.embedding(x)  # [b, seq_len, 15, embedding_dim]
        # 15 ve embedding_dim boyutlarını birleştir: (b, seq_len, 15*embedding_dim)
        x_embed = x_embed.view(batch_size, seq_len, -1)

        # Değişken uzunluklu sequence'leri pack/pad
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_embed,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed_x)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # LSTM çıkışı sonrası dropout
        out = self.post_lstm_dropout(out)

        # Ek FF katmanları
        out = F.relu(self.fc1(out))     # (b, seq_len, hidden_dim)
        out = self.post_lstm_dropout(out)
        out = self.fc2(out)            # (b, seq_len, 5*vocab_size)

        # (b, seq_len, 5, vocab_size) formatına dönüştür
        b_size, seq_len, _ = out.size()
        out = out.view(b_size, seq_len, 5, -1)

        return out

letter2index = {letter: idx for idx, letter in enumerate(
    ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P", "R",
     "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"])}
index2letter = {i: ch for ch, i in letter2index.items()}


def encode_word(word):
    encode = [letter2index[ch] for ch in word]
    return encode


def decode_model_output(encoded_word):
    decode = "".join(index2letter[idx] for idx in encoded_word)
    return decode


def prepare_input(wordleGuesses):
    all_input = []
    for wordleGuess in wordleGuesses.guesses:
        encoded = encode_word(wordleGuess.guess)
        item = encoded + wordleGuess.feedback
        all_input.append(item)
    return all_input


def predict(wordleGuesses):  # input[[]] liste içinde liste şeklinde
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordleLSTM(
        vocab_size=29,  # Harf sayısı
        embedding_dim=16,  # Embedding boyutu
        input_dim=10,  # Girdi boyutu
        hidden_dim=256,  # LSTM gizli boyutu
        num_layers=4,  # LSTM katman sayısı
        dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load("model/LSTMmodel_100epoch.pth", map_location=device))
    model.eval()
    model_input = prepare_input(wordleGuesses=wordleGuesses)
    sequence = torch.tensor(model_input, dtype=torch.long)
    sequence = sequence.unsqueeze(0)
    lengths = torch.tensor([len(model_input)], dtype=torch.long)  # Her dizinin gerçek uzunluğu
    sequence = sequence.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        logits = model(sequence, lengths)  # (1, 3, 5, 29)
        # Son boyut 29 sınıfa karşılık geliyor, argmax ile hangi sınıf olduğu bulunur
        predictions = torch.argmax(logits, dim=-1)  # (1, 3, 5)
        decoded_predictions = []
        for t in range(predictions.shape[1]):  # seq_len = 3
            # Tek bir zaman adımındaki (5) tahmin indekslerini al
            step_preds = predictions[0, t]  # (5,)
            decoded_predictions.append(decode_model_output(step_preds.tolist()))

        # Feedback e göre kelime havuzunu daralt

        remaining_words = filter_possible_words(wordleGuesses.guesses)
        final_prediction = get_word_similartiy(decoded_predictions[-1], remaining_words)
        print("Hibrik tahmin: "+final_prediction)
        return final_prediction


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

def filter_possible_words(guesses):
    possible_words = read_words_from_file()

    for guess, feedback in guesses:
        filtered = []
        for word in possible_words:
            if generate_feedback(word, guess[1]) == tuple(feedback[1]):
                filtered.append(word)
        possible_words = filtered

    return possible_words
def get_word_similartiy(prediction,remaining_words):

    # Kelimeler arasındaki cosine benzerliği hesaplama
    def get_word_similarity(word1, word2):
        vec1 = np.array(word1)
        vec2 = np.array(word2)
        # Vektörlerin ortalamasını alıyoruz
        if len(vec1) == 0 or len(vec2) == 0:
            return 0  # Eğer kelime boşsa benzerlik sıfırdır.
        euclidean_distance = np.linalg.norm(vec1 - vec2)
        return euclidean_distance
    min_similarity = 100000  # Başlangıçta benzerlik değeri en düşük
    most_similar_word = None

    for word in remaining_words:
        similarity = get_word_similarity(encode_word(prediction), encode_word(word))
        if similarity < min_similarity:
            min_similarity = similarity
            most_similar_word = word

    return most_similar_word


def read_words_from_file():
    file_path = "dataset/default_words.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            words = [line.strip().upper() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
        words = []
    finally:
        return words
