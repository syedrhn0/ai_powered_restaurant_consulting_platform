# =============================================================
#   AI-Powered Restaurant Consulting Platform
#   File        : model_building.py
#   Description : Loads data → builds features → trains two
#                 ML models → saves everything to /models
#
#   Run this file ONCE before launching the app:
#       python model_building.py
# =============================================================

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------
# STEP 1 — LOAD DATA
# ------------------------------------------------------------------
print("=" * 55)
print("  AI-Powered Restaurant Consulting Platform")
print("  model_building.py")
print("=" * 55)
print()
print("[ STEP 1 ] Loading datasets...")

restaurants = pd.read_csv("data/restaurants.csv")
customers   = pd.read_csv("data/customers.csv")
orders      = pd.read_csv("data/orders.csv")
reviews     = pd.read_csv("data/reviews.csv")

print(f"  restaurants : {len(restaurants):,} rows  x  {restaurants.shape[1]} cols")
print(f"  customers   : {len(customers):,} rows  x  {customers.shape[1]} cols")
print(f"  orders      : {len(orders):,} rows  x  {orders.shape[1]} cols")
print(f"  reviews     : {len(reviews):,} rows  x  {reviews.shape[1]} cols")

# ------------------------------------------------------------------
# STEP 2 — BASIC CLEANING
# ------------------------------------------------------------------
print()
print("[ STEP 2 ] Cleaning data...")

# Keep only delivered orders for analysis
delivered_orders = orders[orders["order_status"] == "Delivered"].copy()
print(f"  Total orders     : {len(orders):,}")
print(f"  Delivered orders : {len(delivered_orders):,}  ({len(delivered_orders)/len(orders)*100:.1f}%)")

# Convert boolean columns to int
restaurants["has_online_ordering"] = restaurants["has_online_ordering"].astype(int)
restaurants["has_table_booking"]   = restaurants["has_table_booking"].astype(int)
restaurants["is_active"]           = restaurants["is_active"].astype(int)

# Convert order_date to datetime
orders["order_date"] = pd.to_datetime(orders["order_date"])

print("  Boolean columns converted to int (0/1)")
print("  order_date converted to datetime")

# ------------------------------------------------------------------
# STEP 3 — FEATURE ENGINEERING
# ------------------------------------------------------------------
print()
print("[ STEP 3 ] Building features...")

# --- 3a. Average rating per restaurant (from reviews) ---
avg_rating = (
    reviews
    .groupby("restaurant_id")["overall_rating"]
    .mean()
    .reset_index()
    .rename(columns={"overall_rating": "avg_rating"})
)

# --- 3b. Total delivered orders per restaurant ---
order_counts = (
    delivered_orders
    .groupby("restaurant_id")
    .size()
    .reset_index(name="total_orders")
)

# --- 3c. Merge into master table ---
master = restaurants.copy()
master = master.merge(avg_rating,   on="restaurant_id", how="left")
master = master.merge(order_counts, on="restaurant_id", how="left")
master["avg_rating"]   = master["avg_rating"].fillna(0)
master["total_orders"] = master["total_orders"].fillna(0)

# --- 3d. Competition density ---
# How many restaurants of the same cuisine exist in the same area?
competition = (
    master
    .groupby(["area", "cuisine_type"])
    .size()
    .reset_index(name="competition_density")
)
master = master.merge(competition, on=["area", "cuisine_type"], how="left")

# --- 3e. Cuisine popularity score per area ---
# Score = average of (total_orders × avg_rating) for that cuisine in that area
popularity = (
    master
    .groupby(["area", "cuisine_type"])
    .apply(lambda x: (x["total_orders"] * x["avg_rating"]).mean())
    .reset_index(name="cuisine_popularity_score")
)
master = master.merge(popularity, on=["area", "cuisine_type"], how="left")

# --- 3f. Demand-supply gap ---
# Normalize both scores to 0-1, then subtract
scaler = MinMaxScaler()
master["popularity_norm"]   = scaler.fit_transform(master[["cuisine_popularity_score"]])
master["competition_norm"]  = scaler.fit_transform(master[["competition_density"]])
master["demand_supply_gap"] = master["popularity_norm"] - master["competition_norm"]
# Positive = opportunity (high demand, low competition)
# Negative = saturated market

# --- 3g. Average order value per area ---
orders_with_area = delivered_orders.merge(
    restaurants[["restaurant_id", "area"]], on="restaurant_id"
)
avg_order_value = (
    orders_with_area
    .groupby("area")["total_amount"]
    .mean()
    .reset_index(name="avg_order_value")
)
master = master.merge(avg_order_value, on="area", how="left")
master["avg_order_value"] = master["avg_order_value"].fillna(master["avg_order_value"].mean())

print("  Feature: avg_rating               (from reviews table)")
print("  Feature: total_orders             (delivered orders per restaurant)")
print("  Feature: competition_density      (same cuisine count in area)")
print("  Feature: cuisine_popularity_score (orders × rating per cuisine/area)")
print("  Feature: demand_supply_gap        (popularity − competition, normalised)")
print("  Feature: avg_order_value          (avg spend per area)")

# ------------------------------------------------------------------
# STEP 4 — ENCODE CATEGORICAL COLUMNS
# ------------------------------------------------------------------
print()
print("[ STEP 4 ] Encoding categorical columns...")

le_city    = LabelEncoder()
le_area    = LabelEncoder()
le_cuisine = LabelEncoder()
le_type    = LabelEncoder()

master["city_encoded"]    = le_city.fit_transform(master["city"])
master["area_encoded"]    = le_area.fit_transform(master["area"])
master["cuisine_encoded"] = le_cuisine.fit_transform(master["cuisine_type"])
master["type_encoded"]    = le_type.fit_transform(master["restaurant_type"])

print(f"  city    → {le_city.classes_.tolist()}")
print(f"  area    → {le_area.classes_.shape[0]} unique areas encoded")
print(f"  cuisine → {le_cuisine.classes_.tolist()}")
print(f"  type    → {le_type.classes_.tolist()}")

# ------------------------------------------------------------------
# STEP 5 — TRAIN MODEL 1  (Best Location Finder)
# ------------------------------------------------------------------
print()
print("[ STEP 5 ] Training Model 1 — Best Location Finder...")
print("  Input  : cuisine, type, price, online_ordering, table_booking,")
print("           competition_density, cuisine_popularity_score,")
print("           demand_supply_gap, avg_order_value")
print("  Target : area (best area to open the restaurant)")

features_m1 = [
    "cuisine_encoded",
    "type_encoded",
    "avg_price_for_two",
    "has_online_ordering",
    "has_table_booking",
    "competition_density",
    "cuisine_popularity_score",
    "demand_supply_gap",
    "avg_order_value",
]
target_m1 = "area_encoded"

X1 = master[features_m1]
y1 = master[target_m1]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X1_train, y1_train)

acc1 = accuracy_score(y1_test, model1.predict(X1_test))
print(f"  Train size : {X1_train.shape[0]:,} | Test size : {X1_test.shape[0]:,}")
print(f"  Accuracy   : {acc1 * 100:.2f}%")

# ------------------------------------------------------------------
# STEP 6 — TRAIN MODEL 2  (Best Cuisine Finder)
# ------------------------------------------------------------------
print()
print("[ STEP 6 ] Training Model 2 — Best Cuisine Finder...")
print("  Input  : city, area, price, competition_density,")
print("           cuisine_popularity_score, demand_supply_gap, avg_order_value")
print("  Target : cuisine_type (best cuisine to serve)")

features_m2 = [
    "city_encoded",
    "area_encoded",
    "avg_price_for_two",
    "competition_density",
    "cuisine_popularity_score",
    "demand_supply_gap",
    "avg_order_value",
]
target_m2 = "cuisine_encoded"

X2 = master[features_m2]
y2 = master[target_m2]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X2_train, y2_train)

acc2 = accuracy_score(y2_test, model2.predict(X2_test))
print(f"  Train size : {X2_train.shape[0]:,} | Test size : {X2_test.shape[0]:,}")
print(f"  Accuracy   : {acc2 * 100:.2f}%")

# ------------------------------------------------------------------
# STEP 7 — SAVE MODELS & ENCODERS
# ------------------------------------------------------------------
print()
print("[ STEP 7 ] Saving models and encoders to /models ...")

os.makedirs("models", exist_ok=True)

# Save trained models
joblib.dump(model1, "models/location_model.pkl")
joblib.dump(model2, "models/cuisine_model.pkl")

# Save label encoders (needed by app.py to decode predictions back to text)
joblib.dump(le_area,    "models/label_encoder_area.pkl")
joblib.dump(le_cuisine, "models/label_encoder_cuisine.pkl")
joblib.dump(le_city,    "models/label_encoder_city.pkl")
joblib.dump(le_type,    "models/label_encoder_type.pkl")

# Save feature data (app.py uses this for area/city dropdowns and lookup)
feature_data = master[[
    "area", "city", "cuisine_type", "restaurant_type",
    "avg_price_for_two", "competition_density",
    "cuisine_popularity_score", "demand_supply_gap",
    "avg_order_value", "avg_rating", "total_orders",
    "area_encoded", "city_encoded", "cuisine_encoded", "type_encoded",
    "has_online_ordering", "has_table_booking",
]].copy()

joblib.dump(feature_data, "models/feature_data.pkl")

print("  location_model.pkl       ✓")
print("  cuisine_model.pkl        ✓")
print("  label_encoder_area.pkl   ✓")
print("  label_encoder_cuisine.pkl✓")
print("  label_encoder_city.pkl   ✓")
print("  label_encoder_type.pkl   ✓")
print("  feature_data.pkl         ✓")

# ------------------------------------------------------------------
# DONE
# ------------------------------------------------------------------
print()
print("=" * 55)
print("  TRAINING COMPLETE!")
print("=" * 55)
print(f"  Model 1  —  Best Location Finder : {acc1 * 100:.2f}%")
print(f"  Model 2  —  Best Cuisine Finder  : {acc2 * 100:.2f}%")
print()
print("  All files saved to /models")
print()
print("  Next step — launch the app:")
print("  streamlit run app.py")
print("=" * 55)
