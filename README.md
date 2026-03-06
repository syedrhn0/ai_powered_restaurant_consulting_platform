# 🍽️ AI-Powered Restaurant Consulting Platform

> Data Science Capstone Project  
> Python · Machine Learning · Streamlit

---

## 📁 Project Structure

```
restaurant_consulting/
│
├── data/                  ← Raw datasets (4 CSV files)
│   ├── restaurants.csv    (3,000 rows)
│   ├── customers.csv      (10,000 rows)
│   ├── orders.csv         (50,000 rows)
│   └── reviews.csv        (~30,000 rows)
│
├── models/                ← Saved ML models (auto-generated)
│   ├── location_model.pkl
│   ├── cuisine_model.pkl
│   ├── label_encoder_*.pkl
│   └── feature_data.pkl
│
├── model_building.py      ← Loads data · builds features · trains & saves models
├── app.py                 ← Streamlit web application
├── EDA.ipynb              ← Exploratory Data Analysis notebook
├── requirements.txt       ← Python dependencies
└── README.md
```

---

## ⚙️ How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the models
```bash
python model_building.py
```
This reads data from `/data`, trains both ML models, and saves all `.pkl`
files into `/models`. The `/models` folder already ships with pre-trained
files — re-run this only if you change the data or features.

### Step 3 — Launch the web app
```bash
streamlit run app.py
```

### Step 4 — Open in browser
```
http://localhost:8501
```

### Optional — EDA Notebook

Run all cells to see data exploration, 12 charts, and feature engineering walkthrough.

---

## 🤖 ML Models

| Model | Task | Algorithm | Accuracy |
|-------|------|-----------|----------|
| Model 1 | Best Location Finder | Random Forest | ~90.7% |
| Model 2 | Best Cuisine Finder  | Random Forest | ~81.2% |

---

## 🏙️ Cities Covered
Mumbai · Delhi · Bangalore · Hyderabad · Pune · Chennai · Kolkata
