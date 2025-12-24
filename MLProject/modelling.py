"""
modelling.py
------------
Modul ini bertanggung jawab untuk melatih Baseline Model menggunakan Deep Learning (TensorFlow/Keras).
Script ini mencakup pipeline standar ML: Loading -> Splitting -> Scaling -> Training -> Evaluation -> Saving.

Fitur Utama:
1. Arsitektur Deep Neural Network (DNN) dengan Dropout untuk regularisasi.
2. Early Stopping untuk mencegah Overfitting dan menghemat waktu komputasi.
3. Penyimpanan Artifacts (Model .h5 & Scaler .pkl) untuk deployment.
4. Visualisasi Learning Curve (Loss & Accuracy).
5. Integrasi MLflow untuk tracking experiment.

Author: Caleb Anthony (Automated by System)
Date: 2025-10-30
Version: 1.1 (Baseline Deep Learning + MLflow)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import dagshub

# --- KONFIGURASI LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- KONFIGURASI PROJECT ---
class ModelConfig:
    # Path Handling
    BASE_DIR = Path(__file__).resolve().parent
    # Mengambil data bersih dari folder eksperimen (Single Source of Truth)
    # Adjusted path for MLProject execution context
    DATA_PATH = BASE_DIR.parent.parent / 'Eksperimen_SML_CalebAnthony' / 'churn_preprocessing' / 'clean_data.csv'
    
    # Output Artifacts
    ARTIFACTS_DIR = BASE_DIR / 'artifacts'
    MODEL_SAVE_PATH = ARTIFACTS_DIR / 'baseline_model.h5'
    SCALER_SAVE_PATH = ARTIFACTS_DIR / 'scaler.pkl'
    HISTORY_PLOT_PATH = ARTIFACTS_DIR / 'training_history.png'
    
    # Model Parameters (Baseline)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 50          # Batas maksimal epoch
    BATCH_SIZE = 32      # Standar untuk dataset ukuran menengah
    LEARNING_RATE = 0.001
    PATIENCE = 5         # Stop training jika tidak ada perbaikan selama 5 epoch

def parse_args():
    parser = argparse.ArgumentParser(description="Churn Prediction Baseline Model")
    parser.add_argument("--epochs", type=int, default=ModelConfig.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=ModelConfig.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=ModelConfig.LEARNING_RATE, help="Learning rate")
    return parser.parse_args()

class ChurnBaselineTrainer:
    """
    Kelas Trainer untuk Baseline Deep Learning Model.
    """
    
    def __init__(self, epochs, batch_size, learning_rate):
        self.df: Optional[pd.DataFrame] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
        # Override config with args
        ModelConfig.EPOCHS = epochs
        ModelConfig.BATCH_SIZE = batch_size
        ModelConfig.LEARNING_RATE = learning_rate

        # Buat folder artifacts jika belum ada
        ModelConfig.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trainer diinisialisasi. Artifacts akan disimpan di: {ModelConfig.ARTIFACTS_DIR}")

    def load_and_split_data(self) -> None:
        """
        Memuat data, memisahkan fitur & target, melakukan split, dan scaling.
        PENTING: Scaling dilakukan SETELAH split untuk mencegah Data Leakage.
        """
        if not ModelConfig.DATA_PATH.exists():
            logger.critical(f"Data tidak ditemukan di: {ModelConfig.DATA_PATH}")
            sys.exit(1)
            
        logger.info("Memuat dataset...")
        self.df = pd.read_csv(ModelConfig.DATA_PATH)
        
        # Definisi Target
        target_col = 'default'
        if target_col not in self.df.columns:
            raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")
            
        # Pisahkan X (Features) dan y (Target)
        X = self.df.drop(columns=[target_col], errors='ignore')
        y = self.df[target_col]
        
        logger.info(f"Dimensi Fitur: {X.shape}, Target Distribution: {y.value_counts().to_dict()}")

        # Train-Test Split (Stratified agar proporsi churn terjaga)
        logger.info("Membagi data menjadi Train dan Test Set...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=ModelConfig.TEST_SIZE, 
            random_state=ModelConfig.RANDOM_STATE, 
            stratify=y
        )
        
        # Scaling (Standardization)
        logger.info("Melakukan Scaling fitur (StandardScaler)...")
        # Fit hanya pada TRAIN, Transform pada TRAIN dan TEST
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Simpan Scaler (Penting untuk tahap Deployment/Serving nanti)
        joblib.dump(self.scaler, ModelConfig.SCALER_SAVE_PATH)
        mlflow.log_artifact(ModelConfig.SCALER_SAVE_PATH) # Log scaler to MLflow
        logger.info(f"Scaler tersimpan di: {ModelConfig.SCALER_SAVE_PATH}")

    def build_model(self) -> None:
        """
        Membangun arsitektur Neural Network (Multilayer Perceptron).
        Arsitektur: Input -> Dense(64) -> Dropout -> Dense(32) -> Output(Sigmoid)
        """
        input_dim = self.X_train.shape[1]
        logger.info(f"Membangun model Deep Learning dengan input dimension: {input_dim}...")
        
        self.model = Sequential([
            # Hidden Layer 1: Cukup besar untuk menangkap pola
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), # Menstabilkan learning
            Dropout(0.3),         # Mencegah overfitting (mematikan 30% neuron secara acak)
            
            # Hidden Layer 2: Mengerucut
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output Layer: 1 Neuron untuk Binary Classification (0 s/d 1)
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=ModelConfig.LEARNING_RATE)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Wajib untuk binary classification
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model.summary(print_fn=logger.info)

    def train(self) -> None:
        """
        Melatih model dengan Callbacks (EarlyStopping).
        """
        if self.model is None:
            raise ValueError("Model belum dibangun. Jalankan build_model() dulu.")
            
        logger.info("Memulai Training Model...")
        
        # Callbacks
        callbacks = [
            # Berhenti jika val_loss tidak membaik setelah 5 epoch (Patience)
            EarlyStopping(monitor='val_loss', patience=ModelConfig.PATIENCE, restore_best_weights=True),
            # Menyimpan model terbaik selama proses training (checkpointing)
            ModelCheckpoint(filepath=str(ModelConfig.MODEL_SAVE_PATH), monitor='val_loss', save_best_only=True)
        ]
        
        # Enable MLflow autologging
        mlflow.tensorflow.autolog()

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=ModelConfig.EPOCHS,
            batch_size=ModelConfig.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        self._plot_history(history)

    def _plot_history(self, history) -> None:
        """
        Helper function untuk visualisasi kurva Loss dan Accuracy.
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 5))
        
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        plt.tight_layout()
        plt.savefig(ModelConfig.HISTORY_PLOT_PATH)
        mlflow.log_artifact(ModelConfig.HISTORY_PLOT_PATH) # Log plot to MLflow
        plt.close()
        logger.info(f"Grafik training history tersimpan di: {ModelConfig.HISTORY_PLOT_PATH}")

    def evaluate(self) -> None:
        """
        Evaluasi performa model pada Test Set.
        """
        logger.info("Mengevaluasi model pada Test Set...")
        
        # Prediksi Probabilitas
        y_pred_probs = self.model.predict(self.X_test)
        # Thresholding (biasanya 0.5)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Metrics
        print("\n" + "="*50)
        print("Laporan Klasifikasi (Test Set):")
        print(classification_report(self.y_test, y_pred))
        
        auc = roc_auc_score(self.y_test, y_pred_probs)
        print(f"ROC-AUC Score: {auc:.4f}")
        mlflow.log_metric("test_auc", auc)

        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("="*50 + "\n")

    def run(self):
        """
        Orkestrasi seluruh pipeline.
        """
        # Init DagsHub
        dagshub.init(repo_owner='iamikhsank', repo_name='SMSML_CalebAnthony', mlflow=True)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log params
            mlflow.log_param("epochs", ModelConfig.EPOCHS)
            mlflow.log_param("batch_size", ModelConfig.BATCH_SIZE)
            mlflow.log_param("learning_rate", ModelConfig.LEARNING_RATE)
            
            try:
                self.load_and_split_data()
                self.build_model()
                self.train()
                self.evaluate()
                logger.info("=== BASELINE MODELLING SELESAI ===")
            except Exception as e:
                logger.critical(f"Terjadi error fatal: {e}")
                raise

if __name__ == "__main__":
    args = parse_args()
    trainer = ChurnBaselineTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    trainer.run()
