import streamlit as st
import pickle
import os

st.title("Flight Fare Prediction App")

# Load model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model, scaler = load_model()
    st.success("Model Loaded Successfully!")
else:
    st.error("Upload model.pkl and scaler.pkl to this app folder in Streamlit.")

airline = st.selectbox("Airline", ["Vistara", "Air_India", "Indigo", "GO_FIRST", "SpiceJet", "AirAsia"])
source_city = st.selectbox("Source City", ["Delhi", "Bangalore", "Mumbai", "Hyderabad", "Kolkata", "Chennai"])
destination_city = st.selectbox("Destination City", ["Mumbai", "Hyderabad", "Bangalore", "Kolkata", "Chennai", "Delhi"])
departure_time = st.selectbox("Departure Time", ["Morning", "Early_Morning", "Afternoon", "Evening", "Night", "Late_Night"])
arrival_time = st.selectbox("Arrival Time", ["Night", "Morning", "Early_Morning", "Afternoon", "Evening", "Late_Night"])
stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
flight_class = st.selectbox("Flight Class", ["Economy", "Business"])
duration = st.number_input("Duration (hrs)", min_value=0.1)
days_left = st.number_input("Days Left", min_value=1, max_value=365)


if st.button("Predict Price"):

    if os.path.exists(MODEL_PATH):
        # mappings
        airline_mapping = {'Vistara': 0, 'Air_India': 1, 'Indigo': 2, 'GO_FIRST': 3, 'SpiceJet': 4, 'AirAsia': 5}
        source_city_mapping = {'Delhi': 0, 'Bangalore': 1, 'Mumbai': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Chennai': 5}
        destination_city_mapping = {'Mumbai': 0, 'Hyderabad': 1, 'Bangalore': 2, 'Kolkata': 3, 'Chennai': 4, 'Delhi': 5}
        departure_time_mapping = {'Morning': 0, 'Early_Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5}
        arrival_time_mapping = {'Night': 0, 'Morning': 1, 'Early_Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Late_Night': 5}
        stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
        class_mapping = {'Economy': 0, 'Business': 1}

        features = [[
            airline_mapping[airline],
            source_city_mapping[source_city],
            departure_time_mapping[departure_time],
            stops_mapping[stops],
            arrival_time_mapping[arrival_time],
            destination_city_mapping[destination_city],
            class_mapping[flight_class],
            duration,
            days_left
        ]]

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        st.success(f"Predicted Flight Price: ₹ {round(prediction, 2)}")
    else:
        st.error("Model files missing! Please upload model.pkl & scaler.pkl.")
