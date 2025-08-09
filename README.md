# 🚀 VAE-Powered Synthetic Data Generator & Classifier

**Author:** [Neeti Malu](https://github.com/neeti26)  
**Tags:** Variational Autoencoder, Synthetic Data, Network Security, FastAPI, Streamlit

---

## 📌 Overview
This project demonstrates a **Variational Autoencoder (VAE)** based **synthetic data generator** and a **dummy classifier** for handling **imbalanced network intrusion detection datasets**.

It combines:
- **Backend API** → FastAPI (for synthetic data generation & prediction endpoints)
- **Frontend UI** → Streamlit (for interactive data generation & visualization)
- **Research Notebook** → Detailed Jupyter notebook with training, evaluation, and results

---

## 🧠 Architecture

```plaintext
 ┌───────────────┐       API Calls       ┌───────────────┐
 │   Streamlit   │  ─────────────────▶   │    FastAPI    │
 │   Frontend    │                       │   Backend     │
 └───────▲───────┘                       └───────▲───────┘
         │                                  ┌──────┴─────┐
         │                                  │   VAE Model │
         │                                  └─────────────┘
         ▼
     User Input
📂 Project Structure
VAE-Powered-Synthetic-Data-Generator/
│
├── VAE_powered_Synthetic_Data_Generator_for_Imbalanced_Network_Intrusion_Detection.ipynb   # Research & model training
├── app.py             # Streamlit UI
├── main.py            # FastAPI backend
├── requirements.txt   # Dependencies
├── README.md          # Project documentation
└── .gitignore         # Ignore unnecessary files

⚡ Installation & Usage

1️⃣ Clone the repository

git clone https://github.com/neeti26/VAE-Powered-Synthetic-Data-Generator.git
cd VAE-Powered-Synthetic-Data-Generator
2️⃣ Install dependencies

pip install -r requirements.txt
3️⃣ Start the backend (FastAPI)

uvicorn main:app --reload --port 8000
4️⃣ Start the frontend (Streamlit)

streamlit run app.py
