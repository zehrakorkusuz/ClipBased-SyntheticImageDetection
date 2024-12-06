# experiments/predict_clip_svm.py

import torch
import joblib
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from feature_extractor import CLIPFeatureExtractor
from torchvision import transforms
from train_config import TrainConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class CLIPSVMPredictor:
    def __init__(self, model_dir, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.feature_extractor = CLIPFeatureExtractor(device=device)
        
        # Load models
        self.svm = joblib.load(os.path.join(model_dir, 'svm_model.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    

    def predict_single(self, image_path):
        """Predict a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor.extract_features(image)
                features = features.cpu().numpy()
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and probability
            pred = self.svm.predict(features_scaled)[0]
            prob = self.svm.predict_proba(features_scaled)[0]
            
            return pred, prob
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None

    def process_batch(self, image_paths):
        """Process a batch of images efficiently"""
        images = []
        valid_paths = []
        predictions = []
        probabilities = []
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = self.transform(image)
                images.append(image)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
                
        if not images:
            return valid_paths, [], []
            
        # Stack all images into a single tensor
        batch_tensor = torch.stack(images).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor.extract_features(batch_tensor)
            features = features.cpu().numpy()
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions and probabilities
        predictions = self.svm.predict(features_scaled)
        probabilities = self.svm.predict_proba(features_scaled)
        
        return valid_paths, predictions, probabilities

    def predict_validation_set(self, val_csv, base_dir):
        """Predict on validation dataset with metrics"""
        df = pd.read_csv(val_csv)
        
        # Debug print
        print("\nType distribution in validation set:")
        print(df['typ'].value_counts())
        
        # Keep original filenames but add test_set directory
        image_paths = [os.path.join(base_dir, 'test_set', filename) for filename in df['filename']]
        
        # Calculate true labels: 0 for real, 1 for fake
        types = df['typ'].tolist()
        true_labels = [0 if 'real' in typ.lower() else 1 for typ in df['typ']]
        
        all_predictions = []
        all_probabilities = []
        processed_paths = []
        processed_labels = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_labels = true_labels[i:i + self.batch_size]
            paths, preds, probs = self.process_batch(batch_paths)
            
            if paths:
                processed_paths.extend(paths)
                all_predictions.extend(preds)
                all_probabilities.extend(probs)
                processed_labels.extend([true_labels[image_paths.index(path)] for path in paths])
        
        # Create results
        results = []
        for path, pred, prob, true_label, img_type in zip(processed_paths, all_predictions, all_probabilities, processed_labels, [types[image_paths.index(path)] for path in processed_paths]):
            results.append({
            'image_path': path,
            'prediction': 'fake' if pred == 1 else 'real',
            'ground_truth': 'real' if true_label == 0 else 'fake',
            'image_type': img_type,
            'confidence': prob[1] if pred == 1 else prob[0],
            'probabilities': {'real': prob[0], 'fake': prob[1]}
            })
        
        # Debug info
        print(f"\nProcessed {len(processed_paths)} images")
        print(f"Label distribution: {pd.Series(processed_labels).value_counts().to_dict()}")
        
        # Calculate metrics
        if processed_labels and len(np.unique(processed_labels)) > 1:
            metrics = {
                'accuracy': accuracy_score(processed_labels, all_predictions),
                'roc_auc': roc_auc_score(processed_labels, [prob[1] for prob in all_probabilities]),
                'precision_recall_f1': precision_recall_fscore_support(processed_labels, all_predictions, average='binary')[:3]
            }
        else:
            print("Warning: Cannot calculate metrics - insufficient data or single class")
            metrics = {
                'accuracy': 0.0,
                'roc_auc': 0.0,
                'precision_recall_f1': (0.0, 0.0, 0.0)
            }
        
        return results, metrics
    

    def save_predictions_to_csv(self, results, output_path):
        """Save prediction results to CSV with ground truth and type"""
        predictions_df = pd.DataFrame([{
            'filename': os.path.basename(r['image_path']),
            'prediction': r['prediction'],
            'ground_truth': r['ground_truth'],
            'image_type': r['image_type'],
            'confidence': r['confidence'],
            'prob_real': r['probabilities']['real'],
            'prob_fake': r['probabilities']['fake']
        } for r in results])
        
        predictions_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

    def save_metrics_to_csv(self, metrics, output_path):
        """Save metrics to CSV"""
        metrics_df = pd.DataFrame([{
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'precision': metrics['precision_recall_f1'][0],
            'recall': metrics['precision_recall_f1'][1],
            'f1_score': metrics['precision_recall_f1'][2]
        }])
        
        metrics_df.to_csv(output_path, index=False)
        print(f"Metrics saved to: {output_path}")

if __name__ == "__main__":
    config = TrainConfig()
    predictor = CLIPSVMPredictor(config.output_dir)
    
    results, metrics = predictor.predict_validation_set(config.val_csv, config.base_dir)
    

    # Save results to CSV files
    predictions_path = os.path.join(config.output_dir, 'predictions.csv')
    metrics_path = os.path.join(config.output_dir, 'metrics.csv')
    
    predictor.save_predictions_to_csv(results, predictions_path)
    predictor.save_metrics_to_csv(metrics, metrics_path)
    
    print("\nValidation Set Results:")
    print("-" * 50)
    for result in results:
        filename = os.path.basename(result['image_path'])
        print(f"\nImage: {filename}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Real: {result['probabilities']['real']:.4f}, Fake: {result['probabilities']['fake']:.4f}")
    
    print("\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision_recall_f1'][0]:.4f}")
    print(f"Recall: {metrics['precision_recall_f1'][1]:.4f}")
    print(f"F1-Score: {metrics['precision_recall_f1'][2]:.4f}")