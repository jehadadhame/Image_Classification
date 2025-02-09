
import tensorflow as tf
import tensorflow_hub as hub

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from PIL import Image
import json

## i face a problem about vram not enugh so i add this line 
## if this didn't solve it run this before : "export CUDA_VISIBLE_DEVICES=-1" to make it run on the cpu 
## here an example of using the commmand 
## python3 predict.py "./Untitled Folder/orange_dahlia.jpg" multi-tensorflowhub-layers-model1734443097.keras

tf.keras.backend.clear_session()

class Network(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        URL1 = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5"
        URL2 = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
        self.feature_extractor1 = hub.KerasLayer(URL1, trainable=False, input_shape=(224, 224, 3))
        self.feature_extractor2 = hub.KerasLayer(URL2, trainable=False, input_shape=(224, 224, 3))
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))

    def call(self, x):
        x1 = self.feature_extractor1(x)
        x2 = self.feature_extractor2(x)
        x = tf.concat([x1, x2], axis=-1)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x
    
    def get_config(self):
        return {"num_classes": self.num_classes}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
def process_image(image_path):
    """Preprocess the image to fit the model input size."""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0 
    return np.expand_dims(img_array, axis=0)


def predict(image_path, model, top_k=5):
    """Predict the top K classes for the input image using the trained model."""
    img = process_image(image_path)
    predictions = model.predict(img)[0]            
    
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    return top_probs, top_indices

def main():
    parser = argparse.ArgumentParser(description="Flower Class Prediction Script")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("model_path", help="Path to the trained Keras model")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to flower names")

    args = parser.parse_args()
    
    model = load_model(
        args.model_path,
        custom_objects={'Network': Network}
                       )
    
    probs, classes = predict(args.image_path, model, args.top_k)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names[str(c)] for c in classes]
    else:
        class_labels = [str(c) for c in classes]
    
    print("\nTop Predictions:")
    for i in range(len(probs)):
        print(f"{i+1}. {class_labels[i]} - Probability: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
