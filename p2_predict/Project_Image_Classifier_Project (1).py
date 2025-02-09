




get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)




tf.debugging.set_log_device_placement(True)




get_ipython().system('nvidia-smi')





dataset, info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)

training_set = dataset['test']
validation_set = dataset['validation']
test_set = dataset['train']
print(training_set)
print(validation_set)
print(test_set)
info





num_training_examples = len(training_set)
num_validation_examples = len(validation_set)
num_testing_examples = len(test_set)
num_classes = info.features['label'].num_classes
classes = info.features['label'].names

print("number of example in training set : ",num_training_examples)
print("number of example in validation set : ",num_validation_examples)
print("number of example in testing set : ",num_testing_examples)
print("number of classes in the dataset : ",num_classes)
print("classes names : ",classes)




for image, label in training_set.take(3):
    print('image Shape:', image.shape)
    print('image label:', label.numpy())




for image, label in training_set.take(1):
    image = image.numpy()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.show()





import json
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
class_names['0']





for image, label in training_set.take(1):
    image = image.numpy()
    plt.imshow(image)
    plt.title(class_names[str(label.numpy())])
    plt.show()





import tensorflow as tf

image_size = 224
batch_size = 32

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.2, 0.2),

])

def training_format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = data_augmentation(image)
    return image, label

def format_image(image, label):
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

training_batches = (
    training_set
    .shuffle(num_training_examples // 4)
    .map(training_format_image)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

validation_batches = (
    validation_set
    .map(format_image)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

testing_batches = (
    test_set
    .map(format_image)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)




for image_batch, label_batch in training_batches.take(1):
    print(image_batch.shape)
    print(label_batch.shape)
numpy_image = image_batch[0]
plt.imshow(numpy_image)
plt.title(class_names[str(label_batch[0].numpy())])
plt.show()






import tensorflow as tf
import tensorflow_hub as hub

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









EPOCHS = 15
model = Network(num_classes)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):
    history = model.fit(
        training_batches,
        epochs=EPOCHS,
        validation_data=validation_batches,
        callbacks=[early_stopping]
        )





import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()







model.evaluate(testing_batches)





import time
t = time.time()

savedModel_directory = f'./{int(t)}.keras'

model.save(savedModel_directory)




savedModel_directory = f'./multi-tensorflowhub-layers-model{int(t)}.keras'

model.save(savedModel_directory)





loaded_model = tf.keras.models.load_model(
    './multi-tensorflowhub-layers-model1734443097.keras', 
    custom_objects={'Network': Network}
)




loaded_model.summary()





import numpy as np
import tensorflow as tf
from PIL import Image

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()





from PIL import Image

image_path = './Untitled Folder/hard-leaved_pocket_orchid.jpg'

im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()






def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    probs = np.sort(predictions[0])[-top_k:][::-1]
    classes = np.argsort(predictions[0])[-top_k:][::-1]
    classes = [str(c) for c in classes]
    classes = [class_names[str(c)] for c in classes]
    
    return probs, classes





probs, classes = predict(image_path, model, 5)
img = Image.open(image_path)
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

ax1.imshow(img)
ax1.axis('off')
ax1.set_title("Input Image")
y_pos = np.arange(len(classes))
ax2.barh(y_pos, probs, color='blue')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(classes)
ax2.invert_yaxis()  
ax2.set_xlabel("Probability")
ax2.set_title("Top 5 Predictions")
plt.tight_layout()
plt.show()
classes











