import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Model Architecture
# -------------------------
class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StackedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# -------------------------
# Load Model
# -------------------------
input_size = 1
hidden_size = 128
output_size = 1

model = StackedGRU(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("stacked_gru_model.pth", map_location=torch.device('cpu')))
model.eval()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚦 Traffic Flow Prediction using Stacked GRU")

st.write("Enter last 24 traffic values (comma separated):")

user_input = st.text_area("Input Values", "")

if st.button("Predict"):

    try:
        values = list(map(float, user_input.split(",")))

        if len(values) != 24:
            st.error("Please enter exactly 24 values.")
        else:
            scaler = MinMaxScaler()
            values_array = np.array(values).reshape(-1,1)
            scaled_values = scaler.fit_transform(values_array)

            input_tensor = torch.tensor(scaled_values.reshape(1,24,1), dtype=torch.float32)

            with torch.no_grad():
                prediction = model(input_tensor).numpy()

            prediction = scaler.inverse_transform(prediction)

            st.success(f"Predicted Next Hour Traffic: {prediction[0][0]:.2f}")

            # Plot
            plt.figure(figsize=(8,4))
            plt.plot(range(24), values, label="Past 24 Hours")
            plt.scatter(24, prediction[0][0], color="red", label="Prediction")
            plt.legend()
            st.pyplot(plt)

    except:
        st.error("Invalid input format. Enter numbers separated by commas.")