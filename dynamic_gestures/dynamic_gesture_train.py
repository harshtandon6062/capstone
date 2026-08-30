import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# 1. Load Data
X = np.load('X_landmarks.npy') 
y_raw = np.load('y_labels.npy')

encoder = LabelEncoder()
y_int = encoder.fit_transform(y_raw)
y_onehot = to_categorical(y_int)
num_classes = len(encoder.classes_)
np.save('classes.npy', encoder.classes_)

# Balance weights for small datasets
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
class_weights_dict = dict(enumerate(weights))

X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.15, random_state=42)

# 2. Optimized Model Architecture
model =models.Sequential([
    layers.Input(shape=(20, 63)),
    layers.BatchNormalization(), 
    layers.LSTM(64, return_sequences=True, activation='tanh'), 
    layers.Dropout(0.2),
    layers.LSTM(128, return_sequences=False, activation='tanh'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Faster learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Relax the patience for early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=40, # Give it more time to find the solution
    restore_best_weights=True
)

print(f"Training with Augmentation for: {encoder.classes_}")
model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=200, 
    batch_size=4, 
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

model.save('gesture_landmark_model.h5')
print("--- MODEL SAVED WITH EARLY STOPPING ---")