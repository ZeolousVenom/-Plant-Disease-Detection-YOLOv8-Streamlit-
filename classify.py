import numpy as np
from tensorflow.keras.models import load_model

class RiceClassifier:
    def __init__(self, model_path: str, class_names):
        self.model = load_model(model_path)
        self.class_names = class_names

    def predict(self, preprocessed_batch: np.ndarray):
        preds = self.model.predict(preprocessed_batch, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        return label, conf, preds