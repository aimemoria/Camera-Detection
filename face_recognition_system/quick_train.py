#!/usr/bin/env python3
"""Quick training script with better validation strategy"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from D_model_architecture import create_stage_b_model

# Load data
train = np.load('processed/stage_b_train.npz')
test = np.load('processed/stage_b_test.npz')

X_train, y_train = train['X'], train['y']
X_test, y_test = test['X'], test['y']

print(f'Training: {len(X_train)} samples')
print(f'Test: {len(X_test)} samples')
print(f'Classes in train: {np.unique(y_train)}')
print(f'Classes in test: {np.unique(y_test)}')

# Create model
model = create_stage_b_model(num_classes=5)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with 20% of training data as validation
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ],
    verbose=1
)

# Final evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'\n===== Final Test Accuracy: {acc*100:.2f}% =====')

# Predict
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(f'Predictions: {y_pred}')
print(f'True labels: {y_test}')

# Per-class accuracy
for i in range(5):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = np.mean(y_pred[mask] == y_test[mask])
        print(f'Person {i+1}: {class_acc*100:.1f}%')

# Save model
model.save('models/stage_b_final.keras')
print('\nModel saved!')
