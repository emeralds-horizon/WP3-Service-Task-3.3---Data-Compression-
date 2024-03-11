import json

import requests

# URL of the model prediction endpoint
url = "http://localhost:8080/v1/models/model-example:predict"

# Data to be sent for prediction
data = {"instances": [[6.8, 2.8, 4.8, 1.4], [1, 1, 1, 1]]}

# Convert data to JSON
json_data = json.dumps(data)

# Send the POST request to the model server
response = requests.post(url, data=json_data)

# Check if the request was successful
if response.status_code == 200:
    # Get the prediction result
    prediction = response.json()
    print("Prediction:", prediction)
else:
    print("Error:", response.status_code, response.text)
