# experiments/train_clip_svm.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV # if not manual grid search
from sklearn.model_selection import StratifiedKFold, ParameterGrid # if manual grid search
import csv

class CLIPSVMTrainer:
    def __init__(self, feature_extractor, config):
        self.feature_extractor = feature_extractor
        self.config = config
        self.scaler = StandardScaler()
        self.svm = SVC(
            kernel='rbf', 
            probability=True,
            random_state=42,
            class_weight='balanced'
        )

        self.metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        self.best_auc = 0
        self.best_model = None

        os.makedirs(config.output_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate and return relevant metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        return accuracy, precision, recall, f1, auc
    
    def log_metrics(self, metrics):
        """Log metrics to history and print them."""
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
            print(f"{metric.capitalize()}: {value:.4f}")
    
    def save_metrics_to_csv(self, results, folder_path=None):
        if folder_path is None:
            folder_path = self.config.output_dir
        os.makedirs(folder_path, exist_ok=True)
        
        csv_path = os.path.join(folder_path, 'metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['C', 'gamma', 'kernel', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1', 'train_auc', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                params = result['params']
                writer.writerow({
                    'C': params['C'],
                    'gamma': params['gamma'],
                    'kernel': params['kernel'],
                    'train_accuracy': result['train_accuracy'],
                    'train_precision': result['train_precision'],
                    'train_recall': result['train_recall'],
                    'train_f1': result['train_f1'],
                    'train_auc': result['train_auc'],
                    'test_accuracy': result['test_accuracy'],
                    'test_precision': result['test_precision'],
                    'test_recall': result['test_recall'],
                    'test_f1': result['test_f1'],
                    'test_auc': result['test_auc']
                })
        print(f"Metrics saved to {csv_path}")

    # Update extract_batch_features in CLIPSVMTrainer
    def extract_batch_features(self, dataloader):
        features_list = []
        labels_list = []
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Get images and labels from batch
            images = batch['image'].to(self.feature_extractor.device)
            labels = batch['label'].numpy()
            
            # Extract CLIP features
            with torch.no_grad():
                features = self.feature_extractor.extract_features(images)
                features = features.cpu().numpy()
            
            features_list.append(features)
            labels_list.append(labels)
            
        return np.vstack(features_list), np.concatenate(labels_list)


    def train(self, train_loader, val_loader):
        # Extract and scale training features
        print("Extracting training features...")
        X_train, y_train = self.extract_batch_features(train_loader)
        X_train = self.scaler.fit_transform(X_train)
        
        # Extract and scale validation features
        print("Extracting validation features...")
        X_val, y_val = self.extract_batch_features(val_loader)
        X_val = self.scaler.transform(X_val)
        
        # Hyperparameter tuning with manual cross-validation
        param_grid = {
            'C': [0.1, 10],#[0.1, 1, 10, 100],
            'gamma': [0.1, 10], #[1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        print("Starting hyperparameter tuning...")
        skf = StratifiedKFold(n_splits=5)
        results = []
        
        print(f"Evaluating parameters: {param_grid}")
        for params in ParameterGrid(param_grid):
            fold_metrics = []
            print(f"Evaluating parameters: {params}")
            for train_index, test_index in skf.split(X_train, y_train):
                X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
                y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
                
                svm = SVC(probability=True, class_weight='balanced', **params)
                svm.fit(X_fold_train, y_fold_train)
                
                y_fold_pred = svm.predict(X_fold_test)
                y_fold_prob = svm.predict_proba(X_fold_test)[:, 1]
                
                metrics = self.calculate_metrics(y_fold_test, y_fold_pred, y_fold_prob)
                fold_metrics.append(metrics)
                
                print(f"Fold Metrics: Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1: {metrics[3]:.4f}, AUC: {metrics[4]:.4f}")
            
            mean_metrics = np.mean(fold_metrics, axis=0)
            print(f"Mean Metrics: Accuracy: {mean_metrics[0]:.4f}, Precision: {mean_metrics[1]:.4f}, Recall: {mean_metrics[2]:.4f}, F1: {mean_metrics[3]:.4f}, AUC: {mean_metrics[4]:.4f}")
            
            # Train on the entire training set with the current parameters
            svm = SVC(probability=True, class_weight='balanced', **params)
            svm.fit(X_train, y_train)
            
            # Calculate training metrics
            train_pred = svm.predict(X_train)
            train_prob = svm.predict_proba(X_train)[:, 1]
            train_accuracy, train_precision, train_recall, train_f1, train_auc = self.calculate_metrics(y_train, train_pred, train_prob)
            
            # Calculate test metrics
            test_pred = svm.predict(X_val)
            test_prob = svm.predict_proba(X_val)[:, 1]
            test_accuracy, test_precision, test_recall, test_f1, test_auc = self.calculate_metrics(y_val, test_pred, test_prob)
            
            result = {
                'params': params,
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc
            }
            results.append(result)
        
        # Set the best estimator as the SVM model
        best_params = max(results, key=lambda x: x['test_auc'])['params']
        self.svm = SVC(probability=True, class_weight='balanced', **best_params)
        self.svm.fit(X_train, y_train)
        print(f"Best model trained with parameters: {best_params}")
        
        # Calculate and log training metrics for the best model
        train_pred = self.svm.predict(X_train)
        train_prob = self.svm.predict_proba(X_train)[:, 1]
        train_accuracy, train_precision, train_recall, train_f1, train_auc = self.calculate_metrics(y_train, train_pred, train_prob)
        
        train_metrics = {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'auc': train_auc
        }
        print("Training Metrics:")
        self.log_metrics(train_metrics)
        
        # Calculate and log validation metrics for the best model
        val_pred = self.svm.predict(X_val)
        val_prob = self.svm.predict_proba(X_val)[:, 1]
        val_accuracy, val_precision, val_recall, val_f1, val_auc = self.calculate_metrics(y_val, val_pred, val_prob)
        
        val_metrics = {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        }
        print("Validation Metrics:")
        self.log_metrics(val_metrics)
        
        # Check if this is the best model based on AUC
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_model = self.svm
        
        # Save models
        self.save_models()
        
        # Save both training and validation metrics to CSV
        self.save_metrics_to_csv(results, folder_path=self.config.output_dir)
        
        return val_auc
    
    def save_models(self):
        # Save both SVM and scaler for inference
        model_path = os.path.join(self.config.output_dir, 'svm_model.joblib')
        scaler_path = os.path.join(self.config.output_dir, 'scaler.joblib')
        
        joblib.dump(self.svm, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Metrics & Model Performances saved to {self.config.output_dir}")
        
        # Save the best model based on AUC
        if self.best_model is not None:
            best_model_path = os.path.join(self.config.output_dir, 'best_svm_model.joblib')
            joblib.dump(self.best_model, best_model_path)
            print(f"Best model saved to {best_model_path}")


# Main training script
def main():
    from train_config import TrainConfig
    from dataset import CLIPPairDataset, ValidationDataset
    from feature_extractor import CLIPFeatureExtractor
    
    config = TrainConfig()
    print(f"Using train file: {config.train_csv}")
    print(f"Using validation file: {config.val_csv}")
    
    train_dataset = CLIPPairDataset(
        csv_path=config.train_csv,
        base_dir=config.base_dir
    )
    val_dataset = ValidationDataset(
        csv_path=config.val_csv,
        base_dir=config.base_dir
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize feature extractor and trainer
    feature_extractor = CLIPFeatureExtractor(device=config.device)
    trainer = CLIPSVMTrainer(feature_extractor, config)
    
    # Train
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()