# =============================================================
#   AI-Powered Restaurant Consulting Platform
#   File        : app.py
#   Description : Streamlit web application
#
#   Run with:
#       streamlit run app.py
#
#   NOTE: Run model_building.py first to generate the model
#         files inside the /models folder.
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Restaurant Consulting",
    page_icon="🍽️",
    layout="wide"
)

# ------------------------------------------------------------------
# LOAD MODELS & DATA  (cached so they load only once)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    model1       = joblib.load("models/location_model.pkl")
    model2       = joblib.load("models/cuisine_model.pkl")
    le_area      = joblib.load("models/label_encoder_area.pkl")
    le_cuisine   = joblib.load("models/label_encoder_cuisine.pkl")
    le_city      = joblib.load("models/label_encoder_city.pkl")
    le_type      = joblib.load("models/label_encoder_type.pkl")
    feature_data = joblib.load("models/feature_data.pkl")
    return model1, model2, le_area, le_cuisine, le_city, le_type, feature_data

try:
    model1, model2, le_area, le_cuisine, le_city, le_type, feature_data = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown("""
    <h1 style='text-align:center; color:#E84040;'>
        🍽️ AI-Powered Restaurant Consulting Platform
    </h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Data-driven insights to help you open the right restaurant at the right place
    </p>
    <hr>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(
        f"⚠️ Models not found!\n\n"
        f"Please run `python model_building.py` first to train and save the models.\n\n"
        f"Error details: {load_error}"
    )
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/restaurant.png", width=80)
    st.markdown("## About This App")
    st.markdown("""
    This platform uses **Machine Learning** trained on
    Zomato-style data from **7 Indian metro cities**
    to give restaurant business recommendations.

    **Cities covered:**
    - 🏙️ Mumbai · Delhi
    - 🏙️ Bangalore · Hyderabad
    - 🏙️ Pune · Chennai · Kolkata

    **Algorithm:** Random Forest Classifier

    **Dataset:**
    - 3,000 restaurants
    - 50,000 orders
    - ~30,000 reviews
    - Jan 2021 – Dec 2023
    """)
    st.markdown("---")
    st.markdown("*Data Science Capstone Project*")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🗺️  Use Case 1 — Find Best Location",
    "🍜  Use Case 2 — Find Best Restaurant Type",
])

# ==================================================================
# TAB 1 — FIND BEST LOCATION
# ==================================================================
with tab1:
    st.markdown("### 🗺️ Find the Best Location for Your Restaurant")
    st.markdown(
        "Tell us about your restaurant concept and we'll predict the **best areas** to open it."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        cuisine_input = st.selectbox(
            "🍛 What cuisine will you serve?",
            sorted(le_cuisine.classes_)
        )
        rest_type_input = st.selectbox(
            "🏠 What type of restaurant?",
            sorted(le_type.classes_)
        )
        price_input = st.slider(
            "💰 Average price for two (₹)",
            min_value=150, max_value=2000, value=500, step=50
        )

    with col2:
        online_input  = st.radio("📱 Will you offer online ordering?", ["Yes", "No"])
        booking_input = st.radio("📅 Will you offer table booking?",   ["Yes", "No"])
        st.markdown(" ")
        predict_btn1  = st.button("🔍 Find Best Locations", use_container_width=True)

    if predict_btn1:
        with st.spinner("Analysing data..."):

            # Encode inputs
            cuisine_enc = le_cuisine.transform([cuisine_input])[0]
            type_enc    = le_type.transform([rest_type_input])[0]
            online_enc  = 1 if online_input  == "Yes" else 0
            booking_enc = 1 if booking_input == "Yes" else 0

            # Use average feature values for this cuisine
            cuisine_subset = feature_data[feature_data["cuisine_type"] == cuisine_input]
            if len(cuisine_subset) == 0:
                cuisine_subset = feature_data   # fallback to overall averages

            avg_competition = cuisine_subset["competition_density"].mean()
            avg_popularity  = cuisine_subset["cuisine_popularity_score"].mean()
            avg_gap         = cuisine_subset["demand_supply_gap"].mean()
            avg_order_val   = cuisine_subset["avg_order_value"].mean()

            # Build input row for the model
            input_data = pd.DataFrame([{
                "cuisine_encoded":          cuisine_enc,
                "type_encoded":             type_enc,
                "avg_price_for_two":        price_input,
                "has_online_ordering":      online_enc,
                "has_table_booking":        booking_enc,
                "competition_density":      avg_competition,
                "cuisine_popularity_score": avg_popularity,
                "demand_supply_gap":        avg_gap,
                "avg_order_value":          avg_order_val,
            }])

            # Get probabilities for all areas → pick Top 5
            proba        = model1.predict_proba(input_data)[0]
            top5_indices = proba.argsort()[-5:][::-1]
            top5_areas   = le_area.inverse_transform(top5_indices)
            top5_scores  = proba[top5_indices] * 100

        st.markdown("---")
        st.markdown("### ✅ Recommended Locations")

        # ── Top 3 cards ──
        card_cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]

        for i in range(3):
            area_name  = top5_areas[i]
            score      = top5_scores[i]

            # Look up city and stats for this area
            area_info   = feature_data[feature_data["area"] == area_name]
            city_name   = area_info["city"].values[0]   if len(area_info) > 0 else "—"
            avg_rating  = area_info["avg_rating"].mean() if len(area_info) > 0 else 0
            competition = area_info["competition_density"].mean() if len(area_info) > 0 else 0
            gap         = area_info["demand_supply_gap"].mean()    if len(area_info) > 0 else 0

            if gap > 0.1:
                strength = "🟢 Strong Opportunity"
            elif gap > 0:
                strength = "🟡 Moderate Opportunity"
            else:
                strength = "🔴 High Competition"

            with card_cols[i]:
                st.markdown(f"""
                <div style='background:#f9f9f9; border-radius:10px; padding:20px;
                            border:1px solid #ddd; text-align:center;'>
                    <h2>{medals[i]}</h2>
                    <h3 style='color:#E84040;'>{area_name}</h3>
                    <p style='color:gray;'>📍 {city_name}</p>
                    <hr>
                    <p>⭐ Avg Area Rating: <b>{avg_rating:.1f}</b></p>
                    <p>🏪 Competition: <b>{int(competition)}</b> restaurants</p>
                    <p>{strength}</p>
                    <p style='color:#555;'>Model Confidence: <b>{score:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(" ")

        # ── Bar chart: Top 5 areas ──
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#E84040" if i == 0 else "#f9a58c" for i in range(5)]
        ax.barh(top5_areas[::-1], top5_scores[::-1], color=colors[::-1])
        ax.set_xlabel("Model Confidence (%)")
        ax.set_title("Top 5 Recommended Areas")
        ax.axvline(x=top5_scores[0], color="gray", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ==================================================================
# TAB 2 — FIND BEST RESTAURANT TYPE
# ==================================================================
with tab2:
    st.markdown("### 🍜 Find the Best Restaurant Type for Your Location")
    st.markdown(
        "Tell us your location and budget — we'll predict the **best cuisine** to serve."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        city_input = st.selectbox(
            "🏙️ Which city?",
            sorted(le_city.classes_)
        )
        # Filter areas dynamically based on selected city
        city_areas = sorted(
            feature_data[feature_data["city"] == city_input]["area"].unique().tolist()
        )
        area_input = st.selectbox("📍 Which area?", city_areas)

    with col2:
        budget_input = st.slider(
            "💰 Your budget — avg price for two (₹)",
            min_value=150, max_value=2000, value=500, step=50
        )
        st.markdown(" ")
        st.markdown(" ")
        predict_btn2 = st.button("🔍 Find Best Cuisine", use_container_width=True)

    if predict_btn2:
        with st.spinner("Analysing location data..."):

            city_enc = le_city.transform([city_input])[0]
            area_enc = le_area.transform([area_input])[0]

            # Get area-level feature averages
            area_info = feature_data[feature_data["area"] == area_input]
            if len(area_info) == 0:
                area_info = feature_data   # fallback

            avg_competition = area_info["competition_density"].mean()
            avg_popularity  = area_info["cuisine_popularity_score"].mean()
            avg_gap         = area_info["demand_supply_gap"].mean()
            avg_order_val   = area_info["avg_order_value"].mean()

            input_data2 = pd.DataFrame([{
                "city_encoded":             city_enc,
                "area_encoded":             area_enc,
                "avg_price_for_two":        budget_input,
                "competition_density":      avg_competition,
                "cuisine_popularity_score": avg_popularity,
                "demand_supply_gap":        avg_gap,
                "avg_order_value":          avg_order_val,
            }])

            proba2        = model2.predict_proba(input_data2)[0]
            top5_idx      = proba2.argsort()[-5:][::-1]
            top5_cuisines = le_cuisine.inverse_transform(top5_idx)
            top5_scores2  = proba2[top5_idx] * 100

        st.markdown("---")
        st.markdown("### ✅ Recommended Cuisine Types")

        # Sample dishes per cuisine
        SAMPLE_DISHES = {
            "North Indian":          "Butter Chicken, Dal Makhani, Paneer Tikka",
            "South Indian":          "Masala Dosa, Idli Sambar, Vada",
            "Chinese":               "Hakka Noodles, Manchurian, Momos",
            "Mughlai":               "Biryani, Kebabs, Nihari",
            "Biryani":               "Chicken Biryani, Mutton Biryani, Raita",
            "Fast Food":             "Burger, Fries, Pizza, Wraps",
            "Street Food":           "Pani Puri, Bhel Puri, Vada Pav",
            "Continental":           "Pasta, Grilled Chicken, Caesar Salad",
            "Italian":               "Margherita Pizza, Pasta, Tiramisu",
            "Seafood":               "Fish Curry, Prawn Masala, Crab Roast",
            "Maharashtrian":         "Misal Pav, Puran Poli, Poha",
            "Bengali":               "Fish Curry, Rosogolla, Mishti Doi",
            "Tandoor":               "Tandoori Chicken, Seekh Kebab, Fish Tikka",
            "Pizza":                 "Margherita, BBQ Chicken, Veggie Supreme",
            "Desserts & Beverages":  "Gulab Jamun, Ice Cream, Cold Coffee",
        }

        card_cols2 = st.columns(3)
        medals     = ["🥇", "🥈", "🥉"]

        for i in range(3):
            cuisine_name = top5_cuisines[i]
            score        = top5_scores2[i]
            dishes       = SAMPLE_DISHES.get(cuisine_name, "Various dishes")

            area_cuisine_info = feature_data[
                (feature_data["area"]         == area_input) &
                (feature_data["cuisine_type"] == cuisine_name)
            ]
            competition_count = int(area_cuisine_info["competition_density"].mean()) \
                                if len(area_cuisine_info) > 0 else 0
            gap = area_cuisine_info["demand_supply_gap"].mean() \
                  if len(area_cuisine_info) > 0 else 0

            if gap > 0.1:
                opportunity = "🟢 High Demand, Low Competition"
            elif gap > 0:
                opportunity = "🟡 Moderate Opportunity"
            else:
                opportunity = "🔴 Saturated Market"

            with card_cols2[i]:
                st.markdown(f"""
                <div style='background:#f9f9f9; border-radius:10px; padding:20px;
                            border:1px solid #ddd; text-align:center;'>
                    <h2>{medals[i]}</h2>
                    <h3 style='color:#E84040;'>{cuisine_name}</h3>
                    <p style='color:gray; font-size:13px;'>🍽️ {dishes}</p>
                    <hr>
                    <p>🏪 Existing Competition: <b>{competition_count}</b> restaurants</p>
                    <p>{opportunity}</p>
                    <p style='color:#555;'>Model Confidence: <b>{score:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(" ")

        # ── Bar chart: Top 5 cuisines ──
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors2 = ["#2196F3" if i == 0 else "#90CAF9" for i in range(5)]
        ax2.barh(top5_cuisines[::-1], top5_scores2[::-1], color=colors2[::-1])
        ax2.set_xlabel("Model Confidence (%)")
        ax2.set_title(f"Top 5 Cuisines for {area_input}, {city_input}")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Area quick stats ──
        st.markdown("---")
        st.markdown(f"### 📍 Quick Stats — {area_input}, {city_input}")

        area_stats = feature_data[feature_data["area"] == area_input]
        if len(area_stats) > 0:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🏪 Restaurants in Area", len(area_stats))
            s2.metric("⭐ Avg Rating",          f"{area_stats['avg_rating'].mean():.1f}")
            s3.metric("💰 Avg Order Value",     f"₹{area_stats['avg_order_value'].mean():.0f}")
            s4.metric("📦 Total Orders",        f"{int(area_stats['total_orders'].sum()):,}")

# ==================================================================
# TAB 3 — DATA INSIGHTS
