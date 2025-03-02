import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()

target = torch.tensor([1])
print(target)

prediction = torch.tensor([[2.0, 5.0]])
loss = loss_function(prediction, target)
print(loss)

prediction = torch.tensor([[1.5, 1.1]])
loss = loss_function(prediction, target)
print(loss)

loss_function = nn.MSELoss()
prediction = torch.tensor([320_000.0])
target = torch.tensor([300_000.0])
loss = loss_function(prediction, target)
print(loss)