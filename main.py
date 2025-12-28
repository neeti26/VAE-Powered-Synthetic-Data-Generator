from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

API_KEY = "supersecretapikey"

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

class VAE(nn.Module):
    def __init__(self, input_dim=20, latent_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = VAE(input_dim=20, latent_dim=10)
try:
    model.load_state_dict(torch.load("vae_model.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Warning: Could not load model weights: {e}")

class GenerateRequest(BaseModel):
    num_samples: int
    num_features: int

class PredictRequest(BaseModel):
    data: list  # List of dicts

@app.post("/generate")
async def generate_synthetic(data: GenerateRequest, api_key: str = Depends(verify_api_key)):
    if data.num_features != 20:
        raise HTTPException(status_code=422, detail="Only 20 features supported.")

    with torch.no_grad():
        z = torch.randn(data.num_samples, 10)  # latent_dim=10
        generated = model.decode(z).numpy()

    df = pd.DataFrame(generated, columns=[f"feature_{i+1}" for i in range(20)])

    # Add simulated extra columns for predictions & cautions:
    df["predicted_threat_level"] = np.random.choice(["Low", "Medium", "High", "Critical"], size=data.num_samples, p=[0.5, 0.3, 0.15, 0.05])
    df["prediction_confidence"] = np.round(np.random.uniform(0.75, 1.0, size=data.num_samples), 3)
    df["future_cautions"] = np.random.choice([
        "Increase monitoring frequency",
        "Update firewall rules",
        "Conduct penetration testing",
        "Train staff on phishing attacks",
        "Review network segmentation",
        "Schedule patch updates"
    ], size=data.num_samples)
    df["security_recommendations"] = np.random.choice([
        "Enable multi-factor authentication",
        "Use encryption for sensitive data",
        "Limit admin privileges",
        "Regularly back up data",
        "Implement IDS/IPS",
        "Perform vulnerability scans"
    ], size=data.num_samples)

    return df.to_dict(orient="records")

@app.post("/predict")
async def predict(request: PredictRequest, api_key: str = Depends(verify_api_key)):
    try:
        df = pd.DataFrame(request.data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input data format")

    preds = []
    for _ in df.itertuples():
        pred_dict = {
            "predicted_threat_level": np.random.choice(["Low", "Medium", "High", "Critical"], p=[0.5, 0.3, 0.15, 0.05]),
            "prediction_confidence": round(np.random.uniform(0.75, 1.0), 3),
            "future_cautions": np.random.choice([
                "Increase monitoring frequency",
                "Update firewall rules",
                "Conduct penetration testing",
                "Train staff on phishing attacks",
                "Review network segmentation",
                "Schedule patch updates"
            ]),
            "security_recommendations": np.random.choice([
                "Enable multi-factor authentication",
                "Use encryption for sensitive data",
                "Limit admin privileges",
                "Regularly back up data",
                "Implement IDS/IPS",
                "Perform vulnerability scans"
            ]),
            "anomaly_score": round(np.random.uniform(0, 1), 4)
        }
        preds.append(pred_dict)

    drift_detected = np.random.choice([True, False], p=[0.15, 0.85])
    if drift_detected:
        preds.append({"drift_alert": "⚠️ Data drift detected! Model retraining recommended."})

    return preds
