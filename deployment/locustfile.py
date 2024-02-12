from locust import HttpUser, task, between
import json
import random

class MyUser(HttpUser):
    wait_time = between(1, 5)  # Time between requests

    @task(1)
    def index(self):
        self.client.get("/")

    @task(2)
    def predict(self):
        # Prepare a sample request payload
        payload = {
           "fixed_acidity": random.uniform(3.8, 15.9),
            "volatile_acidity": random.uniform(0.08, 1.58),
            "citric_acid": random.uniform(0.0, 1.66),
            "residual_sugar": random.uniform(0.6, 65.8),
            "chlorides": random.uniform(0.009, 0.611),
            "free_sulfur_dioxide": random.uniform(1, 289.0),
            "total_sulfur_dioxide": random.uniform(6.0, 440.0),
            "density": random.uniform(0.987110, 1.038980),
            "pH": random.uniform(2.720, 4.010),
            "sulphates": random.uniform(0.220, 2.000),
            "alcohol": random.uniform(8.0, 14.9),
            "red_wine": random.choice([0, 1])
        }

        headers = {"Content-Type": "application/json"}

        # Send a POST request to the predict endpoint
        response = self.client.post("/predict", data=json.dumps(payload), headers=headers)

        print(response.text)

    @task(3)
    def predict_with_invalid_data(self):
        # Send a POST request with invalid data to trigger a potential error
        invalid_payload = {"invalid_field": "invalid_value"}
        headers = {"Content-Type": "application/json"}

        response = self.client.post("/predict", data=json.dumps(invalid_payload), headers=headers)

        print(response.text)
