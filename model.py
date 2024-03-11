import io
import logging
import os
from typing import Dict, Union

import joblib
import numpy as np
from kserve import InferRequest, Model, ModelServer
from kserve.errors import InvalidInput
from minio import Minio

logging.basicConfig(level=logging.INFO)

MINIO_ACCESS_KEY = os.environ['MINIO_ACCESS_KEY']
MINIO_SECRET_ACCESS_KEY = os.environ['MINIO_SECRET_ACCESS_KEY']
MINIO_API_URL = os.environ['MINIO_API_URL']
MINIO_BUCKET = os.environ['MINIO_BUCKET']
MODEL_PATH = os.environ['MODEL_PATH']


class ModelExample(Model):
    """
    A custom KFServing model for iris classification using sklearn.

    Args:
        name (str): The name of the inference service.

    Attributes:
        model: Sklearn model.

    Methods:
        load(): Loads the model from the object storage.
        predict(request: Dict) -> Dict: Makes predictions based on the input request.

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.ready = False
        self.model = None
        self.minio_client = Minio(MINIO_API_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY)

    def load(self):
        """
        Loads the model from the object storage.
        """
        response = self.minio_client.get_object(MINIO_BUCKET, MODEL_PATH)
        with io.BytesIO(response.data) as f:
            self.model = joblib.load(f)
        self.ready = True

    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> np.ndarray:
        if isinstance(payload, Dict) and "instances" in payload:
            logging.info("Inference request: %s", str(payload))
            inputs = payload["instances"]
            inputs = np.array(inputs)
            return inputs
        else:
            raise InvalidInput(
                "Invalid payload. Payload example: {'instances': [[6.8,  2.8,  4.8,  1.4], [1, 1, 1, 1]]}"
            )

    def predict(self, input_array: np.ndarray, headers: Dict[str, str] = None) -> Dict:
        """
        Makes predictions based on the input request.

        Args:
            request (Dict): The input request containing 'instances' field.

        Returns:
            Dict: A dictionary containing the 'prediction' field with the predicted value.

        """
        result = []
        for array in input_array:
            prediction = self.model.predict(array.reshape(1, -1))
            result.append(prediction[0])
        return {"prediction": str(np.array(result))}


if __name__ == "__main__":
    logging.info("Starting Kserve inference service v7 05/02 12:23")
    model = ModelExample("model-example")
    model.load()
    ModelServer().start([model])
