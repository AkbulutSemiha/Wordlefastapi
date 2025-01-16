import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import pandas as pd


class WordleSequenceDataset(Dataset):
    """
    CSV'den okunan veriyi sequence formatında tutar.
    - gameid: Aynı oyuna ait tahmin dizisi
    - attempt_index: Bu oyunun kaçıncı tahmini
    - pg1..pg5, l1..l5, t1..t5, g1..g5: Tamsayı indeks değerleri (harf vb.)
    """
    def __init__(self, data_frame):
        super().__init__()

        self.sequences = []
        self.labels = []
        self.lengths = []

        grouped = data_frame.groupby("gameid")
        for gameid, group in grouped:
            group_sorted = group.sort_values(by="attempt_index")

            # Girdi X = pg1..pg5, l1..l5, t1..t5 (15 boyutlu vektör)
            X = group_sorted[[
                "pg1", "pg2", "pg3", "pg4", "pg5",
                "l1", "l2", "l3", "l4", "l5",
                "t1", "t2", "t3", "t4", "t5"
            ]].values  # (seq_len, 15)

            # Çıktı Y = g1..g5 (5 boyutlu)
            Y = group_sorted[["g1", "g2", "g3", "g4", "g5"]].values  # (seq_len, 5)

            X_tensor = torch.tensor(X, dtype=torch.long)
            Y_tensor = torch.tensor(Y, dtype=torch.long)

            self.sequences.append(X_tensor)
            self.labels.append(Y_tensor)
            self.lengths.append(X_tensor.size(0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


def collate_fn(batch):
    """
    DataLoader için farklı uzunluktaki sequence'leri düzenler.
    batch: List[(X_seq, Y_seq, seq_len)]
    """
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    padded_X = pad_sequence(sequences, batch_first=True)
    padded_Y = pad_sequence(labels, batch_first=True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_X, padded_Y, lengths_tensor


class WordleLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=29,
        embedding_dim=16,
        input_dim=15,
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


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for X, Y, lengths in dataloader:
        X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
        optimizer.zero_grad()

        logits = model(X, lengths)           # (b, seq_len, 5, 29)
        logits_flat = logits.view(-1, 29)    # (b*seq_len*5, 29)
        Y_flat = Y.view(-1)                 # (b*seq_len*5)

        loss = criterion(logits_flat, Y_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits_flat, dim=1)
        correct_predictions += (predicted == Y_flat).sum().item()
        total_predictions += Y_flat.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_loss, accuracy


def test_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for X, Y, lengths in dataloader:
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)

            logits = model(X, lengths)         # (b, seq_len, 5, 29)
            logits_flat = logits.view(-1, 29)  # (b*seq_len*5, 29)
            Y_flat = Y.view(-1)

            loss = criterion(logits_flat, Y_flat)
            total_loss += loss.item()

            _, predicted = torch.max(logits_flat, dim=1)
            correct_predictions += (predicted == Y_flat).sum().item()
            total_predictions += Y_flat.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = 100.0 * correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_loss, accuracy


def main(csv_path="",
         epochs=5,
         batch_size=4,
         lr=1e-3,
         test_ratio=0.2,
         shuffle=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # CSV'yi oku
    df = pd.read_csv(csv_path)

    # Eksik değer kontrolü
    if df.isnull().any().any():
        raise ValueError("CSV dosyasında eksik değerler var!")

    # Tüm gameid'leri al
    all_gameids = df['gameid'].unique()

    # Train/test için gameid bazında ayır
    train_ids, test_ids = train_test_split(all_gameids, test_size=test_ratio, shuffle=shuffle)

    # Ayrılmış ID'lere göre dataframe'leri oluştur
    train_df = df[df['gameid'].isin(train_ids)]
    test_df = df[df['gameid'].isin(test_ids)]

    # Dataset ve DataLoader
    train_dataset = WordleSequenceDataset(train_df)
    test_dataset = WordleSequenceDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, optimizer, loss tanımı
    model = WordleLSTM(
        vocab_size=29,      # Harf sayısı
        embedding_dim=16,   # Embedding boyutu
        input_dim=15,       # Girdi boyutu
        hidden_dim=128,     # LSTM gizli boyutu
        num_layers=2,       # LSTM katman sayısı
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%")
        print("-" * 50)

    torch.save(model.state_dict(), "wordle_model.pth")

def train_wordle(dataset_path):
    main(
        csv_path=dataset_path,
        epochs=100,
        batch_size=128,
        lr=5e-4,
        test_ratio=0.2,
        shuffle=True
    )
