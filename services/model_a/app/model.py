import numpy as np
from sklearn.linear_model import LogisticRegression
import time


MODEL_VERSION = "1.0.0"


class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self):
        # Simulate training/loading
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression()
        model.fit(X, y)

        self.model = model

    def predict(self, features: list[float]) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start = time.perf_counter()

        prediction = self.model.predict_proba([features])[0][1]

        inference_time = (time.perf_counter() - start) * 1000

        return float(prediction), inference_time