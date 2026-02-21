import torch
import torch.nn as nn
import torch.nn.functional as F

class WordleLSTM(nn.Module):
    """
    LSTM-based neural network for predicting Wordle words.
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
        self.output_dim = 5 * vocab_size

        self.letter_embedding = nn.Embedding(vocab_size, letter_embedding_dim)
        self.feedback_embedding = nn.Embedding(3, feedback_embedding_dim)

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
        batch_size, seq_len, _ = x.size()

        letters = x[:, :, :5]
        feedback = x[:, :, 5:]

        letters_emb = self.letter_embedding(letters)
        feedback_emb = self.feedback_embedding(feedback)

        x_embed = torch.cat([letters_emb, feedback_emb], dim=-1)
        x_embed = x_embed.view(batch_size, seq_len, -1)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_embed,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, (h, c) = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = self.post_lstm_dropout(out)
        out = F.relu(self.fc1(out))
        out = self.post_lstm_dropout(out)
        out = self.fc2(out)

        out = out.view(batch_size, seq_len, 5, -1)
        return out
