import os, shutil, zipfile
import splitfolders
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from google.colab import files

print("TensorFlow version:", tf.__version__)
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print("Using device:", device)



# Remove "Unlabeled" folder
unlabeled_dir = "/content/Skin-Disease-Classification-3/train/Unlabeled"
if os.path.exists(unlabeled_dir):
    shutil.rmtree(unlabeled_dir)

# Re-split ensuring balanced classes
splitfolders.ratio("/content/Skin-Disease-Classification-3/train", output="data", seed=42, ratio=(.7, .2, .1))

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Preprocessing
trainGen = ImageDataGenerator(rescale=1./255)
valGen = ImageDataGenerator(rescale=1./255)
testGen = ImageDataGenerator(rescale=1./255)

trainData = trainGen.flow_from_directory("data/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
valData = valGen.flow_from_directory("data/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
testData = testGen.flow_from_directory("data/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

classList = list(trainData.class_indices.keys())
print("Skin Diseases:", classList)

# Load pretrained MobileNetV2
baseModel = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
baseModel.trainable = False  # Freeze base model

# Add custom layers
x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(classList), activation='softmax')(x)
model = Model(inputs=baseModel.input, outputs=output)

# Compile
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(trainData, validation_data=valData, epochs=10)

# Evaluate
test_loss, test_acc = model.evaluate(testData)
print("Test Accuracy:", test_acc)

# Save model
model.save("mobilenet_skin_disease.h5")

# Predict on custom uploaded image
uploaded = files.upload()
for fn in uploaded.keys():
    img = load_img(fn, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classList[np.argmax(prediction)]
    confidence = np.max(prediction)
    print(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
