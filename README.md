# Attack Dreams

*A creative AI experiment that imagines fictional cyber threats.*

---

## Overview

**Attack Dreams** is a speculative AI project built during the **Maximally Vibe-a-thon**.

Instead of detecting real cyber attacks, this project explores a different idea:  
**What if an AI could imagine cyber threats the way humans imagine worst-case scenarios?**

Using a Variational Autoencoder (VAE), Attack Dreams generates **synthetic, fictional network-style data** that represents imagined attack patterns. The goal is not accuracy or production security, but exploration, intuition, and creative expression.

This project sits somewhere between:
- AI experimentation  
- speculative security  
- creative coding  

---

## What it does

- Trains a **Variational Autoencoder (VAE)** on structured, network-like data
- Generates new synthetic samples by sampling the latent space
- Visualizes these imagined scenarios through an interactive **Streamlit dashboard**
- Presents outputs as *fictional attack dreams*, not real detections

The interface is intentionally styled like a futuristic SOC dashboard to reflect how abstract and overwhelming cyber risk can feel.

---

## How it was built

The project consists of two main parts:

### Backend
- **FastAPI** service
- Handles synthetic data generation
- Loads a trained VAE model built with **PyTorch**

### Frontend
- **Streamlit** app
- Interactive controls for generating and exploring scenarios
- Visualizations to make abstract data feel tangible

The system was designed for fast iteration and creative exploration rather than production optimization.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/neeti26/attack-dreams
cd attack-dreams

2. Install dependencies
pip install -r requirements.txt

3. Run the backend (if applicable)
uvicorn main:app --reload --port 8000

4. Run the frontend
streamlit run app.py

Disclaimer

⚠️ Important

This project is an experimental and speculative AI experiment.

All generated data is synthetic and fictional

Outputs should not be used for real security decisions

The project is intended for creativity, exploration, and learning

Built With

Python

PyTorch

Variational Autoencoders (VAE)

FastAPI

Streamlit

NumPy

Pandas

Hackathon Context

This project was built and refined during a 48-hour hackathon.
The code reflects iterative experimentation and creative exploration rather than polished, enterprise-ready software.

That constraint is part of the project’s identity.
