"""
Deep learning models for DDoS detection
LSTM, GRU, BiLSTM, and LSTM-GRU models
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
import numpy as np

def create_lstm_model(input_shape, name="lstm_model"):
    """Create LSTM model for DDoS detection"""
    model = Sequential(name=name)
    
    model.add(LSTM(
        units=128,
        activation='tanh',
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        input_shape=input_shape
    ))
    
    model.add(LSTM(
        units=64,
        activation='tanh',
        dropout=0.3,
        recurrent_dropout=0.3
    ))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def create_gru_model(input_shape, name="gru_model"):
    """Create GRU model for DDoS detection"""
    model = Sequential(name=name)
    
    model.add(GRU(
        units=128,
        activation='tanh',
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        input_shape=input_shape
    ))
    
    model.add(GRU(
        units=64,
        activation='tanh',
        dropout=0.3,
        recurrent_dropout=0.3
    ))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def create_bilstm_model(input_shape, name="bilstm_model"):
    """Create Bidirectional LSTM model for DDoS detection"""
    model = Sequential(name=name)
    
    model.add(Bidirectional(
        LSTM(
            units=64,
            activation='tanh',
            dropout=0.3,
            recurrent_dropout=0.3
        ),
        input_shape=input_shape
    ))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def create_lstm_gru_model(input_shape, name="lstm_gru_model"):
    """Create hybrid LSTM-GRU model for DDoS detection"""
    model = Sequential(name=name)
    
    model.add(LSTM(
        units=200,
        activation='relu',
        return_sequences=True,
        dropout=0.4,
        recurrent_dropout=0.4,
        input_shape=input_shape
    ))
    
    model.add(GRU(
        units=100,
        activation='relu',
        dropout=0.4,
        recurrent_dropout=0.4
    ))
    
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=256):
    """Train a model and return training history"""
    print(f"Training {model.name}...")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print(f"\nEvaluating {model.name}...")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Confusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    print()
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"Specificity: {specificity:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"F1-Score:  {f1_score:.6f}")
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    }

def save_model(model, path):
    """Save model to file"""
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path):
    """Load model from file"""
    model = tf.keras.models.load_model(path)
    print(f"Model loaded from {path}")
    return model