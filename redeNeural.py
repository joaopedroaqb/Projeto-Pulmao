import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import random
import shutil

# Defina as sementes para reproducibilidade
torch.manual_seed(123)

# Diretórios dos dados
data_dir_train = 'train'
data_dir_test = 'test'

# Transformações para treinamento e teste
transform_train = transforms.Compose(
    [
        transforms.Resize([64, 64]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ]
)

# Carregamento dos dados
train_dataset = datasets.ImageFolder(data_dir_train, transform=transform_train)
test_dataset = datasets.ImageFolder(data_dir_test, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# Classe do modelo
class classificador(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, (3, 3))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=14 * 14 * 64, out_features=256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 5)

    def forward(self, X):
        X = self.pool(self.bnorm(self.activation(self.conv1(X))))
        X = self.pool(self.bnorm(self.activation(self.conv2(X))))
        X = self.flatten(X)
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))

        # Saída
        X = self.output(X)

        return X

# Instancie o modelo
net = classificador()

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.08, momentum=0.)

# Dispositivo para treinamento
device = torch.device('cpu')
device

net.to(device)

# Função de treinamento com as alterações necessárias
def training_loop(loader, epoch, model, optimizer, criterion, clip_value):
    running_loss = 0.
    running_accuracy = 0.

    model.train()  # Certifique-se de que o modelo está em modo de treinamento

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        ps = F.softmax(outputs, dim=1)
        top_p, top_class = ps.topk(k=1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.float))
        running_accuracy += accuracy

        loss = criterion(outputs, labels)
        loss.backward()

        # Clip de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        running_loss += loss.item()
        print('\rÉpoca {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisão {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy), end='\r')

    # Imprimindo os dados referentes a essa época
    print('\rÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss / len(loader), running_accuracy / len(loader)))

# Número de épocas
epochs = 20
clip_value = 1.0  # Define o valor de clip dos gradientes

for epoch in range(epochs):
    # Treino
    print("Treinando")
    training_loop(train_loader, epoch, net, optimizer, criterion, clip_value)
    net.eval()
    # Teste
    print("Validando")
    training_loop(test_loader, epoch, net, optimizer, criterion, clip_value)
    net.train()

# Salvar o modelo treinado
net.eval()
torch.save(net.state_dict(), "checkpoint.pth")
