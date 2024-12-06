# train_model.py

# Import necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math
import matplotlib.pyplot as plt

# Data directories
train_dir = 'data/train'
validation_dir = 'data/validation'

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values to [0, 1]
    rotation_range=20,         # Reduced rotation range to prevent over-distortion
    width_shift_range=0.1,     # Reduced shifts
    height_shift_range=0.1,
    shear_range=0.1,           # Reduced shear
    zoom_range=0.1,            # Reduced zoom
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Fill missing pixels after transformations
)

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),    # Resize images to 150x150
    batch_size=32,             # Number of images per batch
    class_mode='binary'        # Binary classification
)

# Validation data should not be augmented
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create the validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Load the pre-trained VGG16 model and freeze its layers
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False  # Freeze the convolutional base

# Calculate steps per epoch using math.ceil
steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

# Model building
model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),       # Regularization to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with a lower learning rate
model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(learning_rate=1e-4),  # Reduced learning rate
    metrics=['accuracy']
)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=3, 
    min_lr=1e-7
)

callbacks = [early_stopping, reduce_lr]

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,  
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Model evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Plot training and validation accuracy and loss
plt.figure(figsize=(8, 8))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Save the trained model
model.save('model/cat_dog_classifier.keras')
