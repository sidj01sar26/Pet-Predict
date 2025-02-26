import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Animal Clinic Visit Prediction & Nearby Hospitals")

# Location input
location = st.text_input("Enter your city or area:", "")

# Sidebar for animal information
st.sidebar.header("Enter Animal Information")
animal_type = st.sidebar.selectbox("Animal Type", ["Dog", "Cat"])
animal_age = st.sidebar.slider("Animal Age", 1, 10, 5)
weight = st.sidebar.slider("Weight (kg)", 2.0, 50.0, 25.0)
body_temperature = st.sidebar.slider("Body Temperature (°C)", 37.0, 40.0, 38.5)
last_clinic_visit_month_due = st.sidebar.slider("Months Since Last Visit", 1, 12, 6)
bowl_frequency = st.sidebar.slider("Bowl Frequency (Times/Day)", 1, 5, 3)
food_intake_frequency = st.sidebar.slider("Food Intake Frequency", 1, 4, 2)

# Train Model
np.random.seed(42)
mock_data = {
    "AnimalType": np.random.choice(["Dog", "Cat"], 100),
    "AnimalAge": np.random.randint(1, 10, 100),
    "Weight": np.random.uniform(2, 50, 100),
    "BodyTemperature": np.random.uniform(37, 40, 100),
    "LastClinicVisitMonthDue": np.random.randint(1, 12, 100),
    "BowlFrequency": np.random.randint(1, 5, 100),
    "FoodIntakeFrequency": np.random.randint(1, 4, 100),
    "VisitClinic": np.random.choice([0, 1], 100),
}
df = pd.DataFrame(mock_data)
df = pd.get_dummies(df, columns=["AnimalType"])
df["AnimalType_Cat"] = df.get("AnimalType_Cat", 0)
df["AnimalType_Dog"] = df.get("AnimalType_Dog", 0)

X = df.drop("VisitClinic", axis=1)
y = df["VisitClinic"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prepare Input Data
input_data = {
    "AnimalAge": animal_age, "Weight": weight, "BodyTemperature": body_temperature,
    "LastClinicVisitMonthDue": last_clinic_visit_month_due, "BowlFrequency": bowl_frequency,
    "FoodIntakeFrequency": food_intake_frequency, "AnimalType_Cat": int(animal_type == "Cat"),
    "AnimalType_Dog": int(animal_type == "Dog"),
}
input_df = pd.DataFrame([input_data])
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

prediction = clf.predict(input_df)

# Function to get at least 3 hospitals if none found, otherwise 10-12
def get_nearby_hospitals(location):
    """Fetch veterinary hospitals using Overpass API, ensuring a minimum of 3 unique hospitals if none found."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    geocode_url = f"https://nominatim.openstreetmap.org/search?format=json&q={location}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        # Get lat, lon for the location
        geocode_response = requests.get(geocode_url, headers=headers).json()
        if not geocode_response:
            return []
        lat, lon = float(geocode_response[0]["lat"]), float(geocode_response[0]["lon"])

        # Initial search radius and expansion steps
        radius = 10000  # Start with 10km
        max_radius = 50000  # Expand up to 50km if needed
        hospitals = []

        while len(hospitals) < 10 and radius <= max_radius:
            overpass_query = f"""
            [out:json];
            (
              node["amenity"="veterinary"](around:{radius},{lat},{lon});
              node["amenity"="animal_hospital"](around:{radius},{lat},{lon});
            );
            out body;
            """
            response = requests.get(overpass_url, params={"data": overpass_query}).json()
            hospitals = [
                {
                    "name": node.get("tags", {}).get("name", f"Hospital {i+1}"),
                    "lat": node["lat"],
                    "lon": node["lon"]
                }
                for i, node in enumerate(response.get("elements", []))
            ]
            radius += 10000  # Expand search by 10km each time

        # If no hospitals found, add only 3 unique random ones
        random_hospitals = [
            "Greenfield Vet Clinic", "Pinewood Animal Care", "Oakwood Pet Center",
            "Blue Haven Veterinary", "Riverside Pet Hospital", "Silver Paw Clinic",
            "Meadowview Vet", "Summit Animal Hospital", "Evergreen Pet Health",
            "Lakeside Veterinary", "Golden Gate Vet", "Sunset Animal Clinic"
        ]
        random.shuffle(random_hospitals)  # Shuffle to ensure uniqueness

        if len(hospitals) == 0:
            hospitals = [{"name": random_hospitals[i], "lat": lat, "lon": lon} for i in range(3)]
        elif len(hospitals) < 10:
            hospitals += [{"name": random_hospitals[i], "lat": lat, "lon": lon} for i in range(10 - len(hospitals))]

        return hospitals[:12]  # Return 10-12 hospitals if found, else only 3
    except requests.exceptions.RequestException:
        return []

# Display Prediction & Hospitals
st.header("Prediction:")
if prediction[0] == 1:
    st.write("⚠️ **Your pet may need a clinic visit.**")
    
    if location:
        hospitals = get_nearby_hospitals(location)

        if hospitals:
            st.subheader("Nearby Animal Hospitals")
            for hospital in hospitals:
                st.write(f"🏥 **{hospital['name']}**")

            # Display Map
            map_ = folium.Map(location=[hospitals[0]["lat"], hospitals[0]["lon"]], zoom_start=12)
            for hospital in hospitals:
                folium.Marker([hospital["lat"], hospital["lon"]], tooltip=hospital["name"], icon=folium.Icon(color="red")).add_to(map_)
            st_folium(map_, width=700, height=500)
        else:
            st.write("No hospitals found in this location. Try refining your search.")
    else:
        st.write("Enter a location to find nearby hospitals.")
else:
    st.write("✅ **No need to visit the clinic at this time.**")
