import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim=41, latent_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc21 = nn.Linear(64, latent_dim)
        self.fc22 = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Classifier(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

def load_vae_model(path=None, device='cpu'):
    model = VAE()
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_classifier_model(path=None, device='cpu'):
    model = Classifier()
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
