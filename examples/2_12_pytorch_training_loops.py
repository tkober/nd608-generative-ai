import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NumberSumDataset(Dataset):
    def __init__(self, data_range=(1, 10)):
        self.numbers = list(range(data_range[0], data_range[1]))


    def __getitem__(self, index):
        number1 = float(self.numbers[index // len(self.numbers)])
        number2 = float(self.numbers[index % len(self.numbers)])
        return torch.tensor([number1, number2]), torch.tensor([number1 + number2])
    
    def __len__(self):
        return len(self.numbers) ** 2
    

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hiddenLayer = nn.Linear(input_size, 256)
        self.outputLayer = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hiddenLayer(x))
        return self.outputLayer(x)
    
dataset = NumberSumDataset(data_range=(1, 100))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
model = MLP(2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    total_loss = 0.0

    for n_pairs, sums in dataloader:
        predictions = model(n_pairs)
        loss = loss_function(predictions, sums)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print('Epoch {}: Sum of Batch Losses = {:.5f}'.format(epoch, total_loss))


print(model(torch.tensor([3.0, 7.0])))