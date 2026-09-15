import streamlit as st
import pandas as pd
import joblib


# ==========================================
# 1. Load model
# ==========================================

model = joblib.load("random_forest_model.pkl")


# ==========================================
# 2. Page configuration
# ==========================================

st.set_page_config(
    page_title="Astana Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# 3. Main title
# ==========================================

st.title("🏠 Astana Property Price Predictor")

st.write(
    "Estimate the asking price of an apartment in Astana "
    "based on its characteristics using a Random Forest model."
)


# ==========================================
# 4. Sidebar
# ==========================================

st.sidebar.header("Property Characteristics")


# Numerical features

area = st.sidebar.number_input(
    "Total Area (m²)",
    min_value=10.0,
    max_value=500.0,
    value=60.0
)

rooms = st.sidebar.number_input(
    "Number of Rooms",
    min_value=1,
    max_value=10,
    value=2
)

kitchen_area = st.sidebar.number_input(
    "Kitchen Area (m²)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

floor = st.sidebar.number_input(
    "Floor",
    min_value=1,
    max_value=50,
    value=5
)

total_floors = st.sidebar.number_input(
    "Total Floors",
    min_value=1,
    max_value=60,
    value=10
)

ceiling_height = st.sidebar.number_input(
    "Ceiling Height (m)",
    min_value=2.0,
    max_value=6.0,
    value=2.7
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=2026,
    value=2020
)


# Categorical features

district = st.sidebar.selectbox(
    "District",
    [
        "Есильский р-н",
        "Нура р-н",
        "Сарайшык р-н",
        "Алматы р-н",
        "р-н Байконур",
        "Сарыарка р-н"
    ]
)


building_type = st.sidebar.selectbox(
    "Building Type",
    [
        "Кирпичный",
        "Монолитный",
        "Панельный"
    ]
)


condition = st.sidebar.selectbox(
    "Property Condition",
    [
        "Евроремонт",
        "Хорошее",
        "Среднее",
        "Требует ремонта"
    ]
)


bathroom = st.sidebar.selectbox(
    "Bathroom",
    [
        "Совмещенный",
        "Раздельный"
    ]
)


balcony = st.sidebar.selectbox(
    "Balcony",
    [
        "Балкон",
        "Лоджия",
        "Нет"
    ]
)


parking = st.sidebar.selectbox(
    "Parking",
    [
        "Есть",
        "Нет"
    ]
)


furniture = st.sidebar.selectbox(
    "Furniture",
    [
        "Есть",
        "Нет"
    ]
)


flooring = st.sidebar.selectbox(
    "Flooring",
    [
        "Ламинат",
        "Линолеум",
        "Паркет"
    ]
)


security = st.sidebar.selectbox(
    "Security",
    [
        "Есть",
        "Нет"
    ]
)


residential_complex = st.sidebar.text_input(
    "Residential Complex",
    value=""
)


# ==========================================
# 5. Prediction
# ==========================================

if st.sidebar.button("Predict Price"):

    input_data = pd.DataFrame({
        "rooms": [rooms],
        "total_area_m2": [area],
        "kitchen_area_m2": [kitchen_area],
        "floor": [floor],
        "total_floors": [total_floors],
        "ceiling_height": [ceiling_height],
        "building_type": [building_type],
        "year_built": [year_built],
        "condition": [condition],
        "bathroom": [bathroom],
        "balcony": [balcony],
        "parking": [parking],
        "furniture": [furniture],
        "flooring": [flooring],
        "security": [security],
        "residential_complex_name": [residential_complex],
        "district": [district]
    })


    prediction = model.predict(input_data)[0]


    # ==========================================
    # 6. Display prediction
    # ==========================================

    st.success("Prediction completed successfully!")

    st.metric(
        "Estimated Asking Price",
        f"{prediction:,.0f} KZT"
    )

    st.write(f"Estimated value: **{prediction / 1_000_000:.1f} million KZT**"
    )


    # ==========================================
    # 7. Display input
    # ==========================================

    st.subheader("Property Details")

    st.dataframe(
        input_data.T.rename(columns={0: "Value"})
    )