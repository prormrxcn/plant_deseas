import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class PlantDiseaseClassifier:
    def __init__(self, model_path, train_dir=None, img_size=(160, 160)):
        
        # Verify and load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Get class names from original dataset
        self.class_names = self._get_class_names(train_dir)
        self.img_size = img_size
        
        # Verify model compatibility
        self._verify_compatibility()

    def _get_class_names(self, train_dir):
        """Extract class names from original dataset directory"""
        if train_dir and os.path.exists(train_dir):
            class_names = sorted([name for name in os.listdir(train_dir) 
                               if os.path.isdir(os.path.join(train_dir, name))])
            if not class_names:
                raise ValueError("No classes found in training directory")
            return class_names
        raise ValueError("Training directory required for class names")

    def _verify_compatibility(self):
        """Verify model matches expected specifications"""
        # Check input shape
        expected_shape = (*self.img_size, 3)
        if self.model.input_shape[1:] != expected_shape:
            raise ValueError(f"Model expects {expected_shape} input, got {self.model.input_shape[1:]}")
            
        # Check output matches class count
        if len(self.class_names) != self.model.output_shape[-1]:
            raise ValueError(f"Model expects {self.model.output_shape[-1]} classes, found {len(self.class_names)}")

    def _preprocess_image(self, image_path):
        """Preprocess image identical to training pipeline"""
        img = image.load_img(image_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path, top_k=3):
        """
        Make prediction on a new image
        :param image_path: Path to image file
        :param top_k: Number of top predictions to return
        :return: Dictionary with predictions and metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Preprocess and predict
        processed_img = self._preprocess_image(image_path)
        predictions = self.model.predict(processed_img)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        return {
            "image": os.path.basename(image_path),
            "predictions": [{
                "class": self.class_names[i],
                "confidence": float(probabilities[i]),
                "class_id": int(i)
            } for i in top_indices],
            "model_info": {
                "input_shape": self.img_size,
                "classes": self.class_names,
                "model_name": os.path.basename(self.model.path)
            }
        }

    def batch_predict(self, image_dir):
        """Predict for all images in a directory"""
        results = []
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_file)
                try:
                    results.append(self.predict(img_path))
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        return results

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    CONFIG = {
        "model_path": "/home/prormrxcn/Documents/plant_disease_model.keras",
        "train_dir": "/home/prormrxcn/Documents/internal_hackathon/train",
        "test_image": "/home/prormrxcn/Documents/360_F_789647578_C1jkPmVBFeJHvi6uuLlFqBSKV9fuBK27.jpg",
    }

    try:
        # Initialize classifier
        classifier = PlantDiseaseClassifier(
            model_path=CONFIG["model_path"],
            train_dir=CONFIG["train_dir"]
        )
        
        print("Model loaded successfully")
        print(f"Input shape: {classifier.model.input_shape}")
        print(f"Classes ({len(classifier.class_names)}): {classifier.class_names}\n")

        # Single image prediction
        print(f"Testing single image: {CONFIG['test_image']}")
        single_result = classifier.predict(CONFIG["test_image"])
        print("\nTop Predictions:")
        for pred in single_result["predictions"]:
            print(f"{pred['class']}: {pred['confidence']:.2%}")

        # Batch prediction (optional)
        if "test_dir" in CONFIG:
            print(f"\nBatch predicting for: {CONFIG['test_dir']}")
            batch_results = classifier.batch_predict(CONFIG["test_dir"])
            print(f"\nProcessed {len(batch_results)} images")

    except Exception as e:
        print(f"Error: {str(e)}")