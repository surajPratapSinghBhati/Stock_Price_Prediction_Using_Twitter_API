# Stock_Price_Prediction_Using_Twitter_API
# AI & ML Project

This repository contains two academic projects demonstrating practical work in AI/ML:

- `stock-sentiment` — Stock prediction combined with Twitter sentiment analysis (Python backend + simple frontend)

---

## Project 1 — Stock Prediction & Sentiment Analysis

**Location:** `stock-sentiment/`

**Summary:**  
Fetches tweets for a query, computes a TextBlob sentiment score, downloads stock data from Yahoo Finance, prepares features (Open, High, Low, Volume, Change, Sentiment) and trains multiple ML models (Linear Regression, KNN, Random Forest, XGBoost, Decision Tree) to predict next-day price direction.

**Key files:**
- `backend.py` — main script to fetch tweets, compute sentiment, download stock data, prepare dataset and train models.
- `frontend/index.html` — demo UI to show tweets, sentiment, stock data and charts.
- `requirements.txt` — Python dependencies.

**How to run (local):**
1. Create virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r stock-sentiment/requirements.txt
