# Import necessary libraries for data analysis and model deployment
import numpy as np
import pandas as pd
import joblib
import gradio as gr



# loading the final model from the disk
cancellation_predictor = joblib.load('../models/hotel_cancellation_prediction_model_v1_0.joblib')

# Define a function to predict booking cancellation
def predict_cancellation(lead_time, market_segment_type, avg_price_per_room, no_of_adults, 
                         no_of_weekend_nights, no_of_week_nights, no_of_special_requests, arrival_month, required_car_parking_space):

    # Prepare the input data as a dictionary
    input_data = {
        'lead_time': lead_time,
        'market_segment_type': 1 if market_segment_type == 'Online' else 0,
        'no_of_special_requests': no_of_special_requests,
        'avg_price_per_room': avg_price_per_room,
        'no_of_adults': no_of_adults,
        'no_of_weekend_nights': no_of_weekend_nights,
        'required_car_parking_space': 1.0 if required_car_parking_space == "Yes" else 0.0,
        'no_of_week_nights': no_of_week_nights,
        'arrival_month': arrival_month,
        
    }

    # Convert input data to a DataFrame
    data_point = pd.DataFrame([input_data])

    # Predict cancellation and its probability
    prediction = cancellation_predictor.predict(data_point).tolist()
    prediction_prob = np.round(100*cancellation_predictor.predict_proba(data_point)[0][0], 2) if prediction == 1 else np.round(100*cancellation_predictor.predict_proba(data_point)[0][1], 2)

    # Return the result as ("Yes", probability) or ("No", probability)
    return ("Yes", str(prediction_prob)+"%") if prediction[0] == 1 else ("No", str(prediction_prob)+"%")

# Define the input interface for Gradio
model_inputs = [
    gr.Number(label="Lead Time (in days)"),
    gr.Dropdown(label="Market Segment Type", choices=["Online", "Offline"]),
    gr.Number(label="Average Price per Room"),
    gr.Number(label="Number of Adults"),
    gr.Number(label="Number of Weekend Nights"),
    gr.Number(label="Number of Week Nights"),
    gr.Number(label="Number of Special Requests"),
    gr.Dropdown(label="Arrival Month", choices=np.arange(1,13,1).tolist()),
    gr.Dropdown(label="Required Car Parking Space", choices=["Yes", "No"])
]

# Define the output interface for Gradio
model_outputs = [
    gr.Textbox(label="Will the booking be cancelled?"),
    gr.Textbox(label="Chances of Cancellation")
]

# Create and configure the Gradio interface
demo = gr.Interface(
    fn = predict_cancellation,
    inputs = model_inputs,
    outputs = model_outputs,
    allow_flagging='never',
    title = "Hotel Booking Cancellation Predictor",
    description = "This interface will predict whether a given hotel booking is likely to be cancelled based on the details of the booking.",
)

# Deploy the Gradio app
demo.launch(inline=False, share=True, debug=True)

# Shut down the deployed model
demo.close()
