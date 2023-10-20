import os
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(123)

class classificador(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) 
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=14*14*64, out_features=256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 5)

    def forward(self, X):
        X = self.pool(self.bnorm(self.activation(self.conv1(X))))
        X = self.pool(self.bnorm(self.activation(self.conv2(X))))
        X = self.flatten(X)

        # Camadas densas
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))
        
        # Saída
        X = self.output(X)

        return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

classificadorLoaded = classificador()
state_dict = torch.load('checkpoint.pth')
classificadorLoaded.load_state_dict(state_dict)


def classificarImagem(nome):
    from PIL import Image
    imagem_teste = Image.open(nome)
    import numpy as np
    imagem_teste = imagem_teste.resize((64, 64))
    imagem_teste = imagem_teste.convert('RGB') 
    imagem_teste = np.array(imagem_teste.getdata()).reshape(*imagem_teste.size, -1)
    imagem_teste = imagem_teste / 255
    imagem_teste = imagem_teste.transpose(2, 0, 1)
    imagem_teste = torch.tensor(imagem_teste, dtype=torch.float).view(-1, *imagem_teste.shape)
    classificadorLoaded.eval()
    imagem_teste = imagem_teste.to(device)
    output = classificadorLoaded.forward(imagem_teste)
    output = F.softmax(output, dim=1)
    output = output.detach().numpy()
    resultado = np.argmax(output[0])
    doencas = ['Atelectasis', 'Effusion', 'Infiltration','Nodule', 'Normal']
    print('Nome Imagem: ' + nome.split('\\')[2] + f' Resultado: {doencas[resultado]} Real: '+ nome.split('\\')[1])

pasta = "val"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
print(caminhos)
for caminho in caminhos:
    file_list = os.listdir(caminho)
    for imagem in file_list:
        classificarImagem(caminho + "\\" + imagem)