import torch

class TunableAttentionRegression(torch.nn.Module):
    def __init__(self, input_size = 2707, hidden_size = 512,
                output_size = 1, numberOfHeads = 1) -> None:
        super(TunableAttentionRegression, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, 32)
        self.lstm = torch.nn.LSTM(32, hidden_size, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=numberOfHeads)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        embedded = self.embedding(x)
        outputList = []
        for eachSequence in range(0, list(embedded.shape)[0]):
            iteratedTensor = embedded[eachSequence, :, :]
            lstm_out, _ = self.lstm(iteratedTensor)
            lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch, hidden_size]
            attention_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            output = self.fc(attention_output.mean(dim=0))
            output = self.sigmoid(output)
            outputList.append(output)

        return torch.stack(outputList)