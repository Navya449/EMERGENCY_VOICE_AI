import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

class EmergencyKWS:
    def __init__(self, input_shape=(49, 64), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self):
         model = keras.Sequential([
            layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1), name='input'),
            layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same', activation='relu', name='conv1'),
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv2'),
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
         return model
    def compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print("\n" + "="*60)
        print("TRAINING MODEL (3 Classes: help, save_me, noise)")
        print("="*60)
        
        if self.model is None:
            self.model = self.build_model()
            self.compile_model()
        
        self.model.summary()
        
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Estimated float32 size: {total_params * 4 / 1024:.1f} KB")
        print(f"Estimated int8 size: {total_params * 1 / 1024:.1f} KB")
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            keras.callbacks.ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        best_epoch = np.argmax(self.history.history['val_accuracy']) + 1
        best_val_acc = np.max(self.history.history['val_accuracy'])
        print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return self.history
    
    def evaluate(self, X_test, y_test, label_map):
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        predictions = self.model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        index_to_label = {v: k for k, v in label_map.items()}
        
        print("\nPer-class accuracy:")
        for class_idx in range(self.num_classes):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(pred_classes[class_mask] == class_idx)
                print(f"   {index_to_label[class_idx]:12s}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
        
        cm = confusion_matrix(y_test, pred_classes)
        print("\nConfusion Matrix:")
        print("   Actual \\ Predicted")
        header = "   " + "".join([f"{index_to_label[i]:>10s}" for i in range(self.num_classes)])
        print(header)
        for i in range(self.num_classes):
            row = f"   {index_to_label[i]:10s}" + "".join([f"{cm[i,j]:10d}" for j in range(self.num_classes)])
            print(row)
        
        return test_accuracy


def main():
    print("Loading preprocessed data...")
    
    X_train = np.load('models/X_train.npy')
    y_train = np.load('models/y_train.npy')
    X_val = np.load('models/X_val.npy')
    y_val = np.load('models/y_val.npy')
    X_test = np.load('models/X_test.npy')
    y_test = np.load('models/y_test.npy')
    
    with open('models/label_map.pkl', 'rb') as f:
        label_data = pickle.load(f)
    label_map = label_data['label_to_index']
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Classes: {list(label_map.keys())}")
    
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    kws = EmergencyKWS(input_shape=(49, 64), num_classes=len(label_map))
    kws.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    kws.evaluate(X_test, y_test, label_map)
    
    print("\nSaving models...")
    kws.model.save('models/final_model.h5')
    tf.saved_model.save(kws.model, 'models/final_model_savedmodel')
    print("Models saved!")


if __name__ == "__main__":
    main()