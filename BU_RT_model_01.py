import torch
import pytorch_lightning as pl

class TunableAttentionRegression(torch.nn.Module):
    def __init__(self, input_size = 223, hidden_size = 64,
                output_size = 1, numberOfHeads = 16) -> None:
        super(TunableAttentionRegression, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, 32)
        self.lstm = torch.nn.LSTM(32, hidden_size, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=numberOfHeads, batch_first=True) #https://pytorch.org/docs/stable/generated/torch.ao.nn.quantizable.MultiheadAttention.html#multiheadattention
        self.linear1 = torch.nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x) -> torch.Tensor:
        x = x.view(x.size(0), x.size(2)*2)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        query = lstm_out.permute(0, 1, 2)
        key = lstm_out.permute(0, 1, 2)
        value = lstm_out.permute(0, 1, 2)
        attention_output, _ = self.attention(query, key, value)
        attention_output_as_2d = attention_output.reshape(attention_output.size(0), attention_output.size(1)*attention_output.size(2))
        output = self.linear1(attention_output_as_2d)
        return output
    

if __name__ == '__main__':
    model = TunableAttentionRegression()
    print(model)