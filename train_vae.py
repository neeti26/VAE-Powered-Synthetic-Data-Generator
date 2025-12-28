import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=10, latent_dim=5):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc21 = nn.Linear(64, latent_dim)
        self.fc22 = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Load and preprocess data
def load_data(path, max_samples=10000):
    df = pd.read_csv(path)
    # Select numeric columns only (adjust columns as needed)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 10:
        raise ValueError("Not enough numeric features. Adjust dataset or preprocessing.")
    numeric_df = numeric_df.iloc[:, :10]  # Use first 10 numeric columns for input
    
    # Normalize features to [0,1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(numeric_df.values[:max_samples])
    
    return torch.tensor(data_scaled, dtype=torch.float32), scaler

def train_vae(data, epochs=30, batch_size=64, learning_rate=1e-3):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE(input_dim=data.shape[1], latent_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x)
            loss = vae_loss(recon_batch, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(data):.6f}")

    return model

if __name__ == "__main__":
    # Change the path below to your extracted NSL-KDD CSV file location
    data_path = "C:/Users/NEETI/Desktop/NSL-KDD/KDDTrain+.csv"


    print("Loading data...")
    data_tensor, scaler = load_data(data_path)

    print("Training VAE...")
    trained_vae = train_vae(data_tensor)

    print("Saving model weights to vae_model.pth")
    torch.save(trained_vae.state_dict(), "vae_model.pth")

    print("Training complete!")