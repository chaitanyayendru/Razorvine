"""
Advanced Predictive Modeling Module
Implements sophisticated ML models for customer analytics with ensemble methods and deep learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostRegressor, CatBoostClassifier

# Deep learning
import tensorflow as tf
import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Feature engineering
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures, DropCorrelatedFeatures

# Model interpretation
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Monitoring and logging
import mlflow
import mlflow.sklearn
import mlflow.keras


class AdvancedPredictiveModeler:
    """
    Sophisticated predictive modeling system with ensemble methods and deep learning.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str, 
                 problem_type: str = 'regression', test_size: float = 0.2):
        """
        Initialize the predictive modeler.
        
        Args:
            data: Input DataFrame
            target_col: Target variable column
            problem_type: 'regression' or 'classification'
            test_size: Proportion of data for testing
        """
        self.data = data.copy()
        self.target_col = target_col
        self.problem_type = problem_type
        self.test_size = test_size
        
        # Initialize MLflow with Windows-compatible path
        try:
            # Use local file system instead of SQLite for better Windows compatibility
            mlflow.set_tracking_uri("file:./mlruns")
        except Exception as e:
            print(f"⚠️ MLflow initialization warning: {e}")
            # Disable MLflow if it fails
            mlflow.set_tracking_uri(None)
        
        # Store models and results
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data for modeling."""
        # Handle missing values
        self.data = self.data.dropna(subset=[self.target_col])
        
        # Separate features and target
        self.features = self.data.drop(columns=[self.target_col])
        self.target = self.data[self.target_col]
        
        # Identify categorical and numerical features
        self.categorical_features = self.features.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = self.features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Convert categorical features to numeric for modeling
        for col in self.categorical_features:
            try:
                self.features[col] = pd.Categorical(self.features[col]).codes
            except:
                # If conversion fails, use simple encoding
                self.features[col] = pd.factorize(self.features[col])[0]
        
        # Create preprocessing pipeline
        self._create_preprocessing_pipeline()
        
        # Split data
        if self.problem_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.features, self.target, test_size=self.test_size, 
                random_state=42, stratify=self.target
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.features, self.target, test_size=self.test_size, random_state=42
            )
        
        # Apply preprocessing
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Convert to DataFrame for feature names
        self.X_train_processed = pd.DataFrame(
            self.X_train_processed,
            columns=self.preprocessor.get_feature_names_out(),
            index=self.X_train.index
        )
        self.X_test_processed = pd.DataFrame(
            self.X_test_processed,
            columns=self.preprocessor.get_feature_names_out(),
            index=self.X_test.index
        )
        
    def _create_preprocessing_pipeline(self):
        """Create sophisticated preprocessing pipeline."""
        # Numerical features pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Categorical features pipeline - simplified to avoid type issues
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop'
        )
        
    def train_ensemble_models(self, models_config: Optional[Dict] = None) -> Dict:
        """
        Train ensemble models with advanced configurations.
        
        Args:
            models_config: Configuration for different models
            
        Returns:
            Dictionary with trained models and results
        """
        if models_config is None:
            models_config = {
                'random_forest': {'n_estimators': 200, 'max_depth': 15, 'random_state': 42},
                'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6},
                'xgboost': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6},
                'lightgbm': {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6},
                'catboost': {'iterations': 200, 'learning_rate': 0.1, 'depth': 6}
            }
        
        print("🚀 Training Ensemble Models...")
        
        results = {}
        
        for model_name, config in models_config.items():
            print(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_{self.problem_type}"):
                # Create model
                if self.problem_type == 'regression':
                    if model_name == 'random_forest':
                        model = RandomForestRegressor(**config)
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingRegressor(**config)
                    elif model_name == 'xgboost':
                        model = xgb.XGBRegressor(**config)
                    elif model_name == 'lightgbm':
                        model = lgb.LGBMRegressor(**config)
                    elif model_name == 'catboost':
                        model = CatBoostRegressor(**config, verbose=False)
                else:
                    if model_name == 'random_forest':
                        model = RandomForestClassifier(**config)
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingClassifier(**config)
                    elif model_name == 'xgboost':
                        model = xgb.XGBClassifier(**config)
                    elif model_name == 'lightgbm':
                        model = lgb.LGBMClassifier(**config)
                    elif model_name == 'catboost':
                        model = CatBoostClassifier(**config, verbose=False)
                
                # Train model
                model.fit(self.X_train_processed, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test_processed)
                y_pred_proba = model.predict_proba(self.X_test_processed) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': self.X_train_processed.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    importance = None
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'predictions_proba': y_pred_proba,
                    'metrics': metrics,
                    'feature_importance': importance
                }
                
                # Log to MLflow
                try:
                    mlflow.log_metrics(metrics)
                    if importance is not None:
                        mlflow.log_artifact(importance.to_csv(), f"{model_name}_feature_importance.csv")
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
                except Exception as e:
                    print(f"⚠️ MLflow logging warning for {model_name}: {e}")
        
        self.models.update(results)
        return results
    
    def train_deep_learning_model(self, architecture: str = 'deep', 
                                 epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train deep learning model with advanced architecture.
        
        Args:
            architecture: Model architecture ('simple', 'deep', 'wide_deep')
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with trained model and results
        """
        print("🧠 Training Deep Learning Model...")
        
        with mlflow.start_run(run_name=f"deep_learning_{self.problem_type}"):
            # Create model architecture
            model = self._create_deep_learning_architecture(architecture)
            
            # Compile model
            if self.problem_type == 'regression':
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
            else:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Callbacks
            callbacks_list = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # Train model
            history = model.fit(
                self.X_train_processed, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Make predictions
            y_pred = model.predict(self.X_test_processed)
            if self.problem_type == 'classification':
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_pred_proba = y_pred
            else:
                y_pred_classes = y_pred.flatten()
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred_classes, y_pred_proba)
            
            # Store results
            results = {
                'model': model,
                'history': history,
                'predictions': y_pred_classes,
                'predictions_proba': y_pred_proba,
                'metrics': metrics
            }
            
            # Log to MLflow
            try:
                mlflow.log_metrics(metrics)
                mlflow.keras.log_model(model, "deep_learning_model")
            except Exception as e:
                print(f"⚠️ MLflow logging warning for deep learning: {e}")
            
            self.models['deep_learning'] = results
            return results
    
    def _create_deep_learning_architecture(self, architecture: str) -> Model:
        """Create deep learning model architecture."""
        input_dim = self.X_train_processed.shape[1]
        
        if architecture == 'simple':
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1 if self.problem_type == 'regression' else len(np.unique(self.y_train)), 
                      activation='linear' if self.problem_type == 'regression' else 'softmax')
            ])
        
        elif architecture == 'deep':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1 if self.problem_type == 'regression' else len(np.unique(self.y_train)), 
                      activation='linear' if self.problem_type == 'regression' else 'softmax')
            ])
        
        elif architecture == 'wide_deep':
            # Wide path (linear)
            wide_input = Input(shape=(input_dim,))
            wide_output = Dense(1, activation='linear')(wide_input)
            
            # Deep path
            deep_input = Input(shape=(input_dim,))
            deep_output = Dense(128, activation='relu')(deep_input)
            deep_output = BatchNormalization()(deep_output)
            deep_output = Dropout(0.4)(deep_output)
            deep_output = Dense(64, activation='relu')(deep_output)
            deep_output = BatchNormalization()(deep_output)
            deep_output = Dropout(0.3)(deep_output)
            deep_output = Dense(32, activation='relu')(deep_output)
            deep_output = Dense(1, activation='linear')(deep_output)
            
            # Combine wide and deep
            combined = layers.Concatenate()([wide_output, deep_output])
            output = Dense(1 if self.problem_type == 'regression' else len(np.unique(self.y_train)), 
                          activation='linear' if self.problem_type == 'regression' else 'softmax')(combined)
            
            model = Model(inputs=[wide_input, deep_input], outputs=output)
        
        return model
    
    def create_ensemble(self, models_to_ensemble: List[str], 
                       ensemble_method: str = 'voting') -> Dict:
        """
        Create ensemble of trained models.
        
        Args:
            models_to_ensemble: List of model names to ensemble
            ensemble_method: 'voting' or 'stacking'
            
        Returns:
            Dictionary with ensemble model and results
        """
        print("🎯 Creating Model Ensemble...")
        
        # Get trained models
        trained_models = []
        for model_name in models_to_ensemble:
            if model_name in self.models:
                trained_models.append(self.models[model_name]['model'])
        
        if not trained_models:
            raise ValueError("No trained models found for ensemble")
        
        # Create ensemble
        if ensemble_method == 'voting':
            if self.problem_type == 'regression':
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in zip(models_to_ensemble, trained_models)],
                    weights=None
                )
            else:
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in zip(models_to_ensemble, trained_models)],
                    voting='soft'
                )
        
        # Train ensemble
        ensemble.fit(self.X_train_processed, self.y_train)
        
        # Make predictions
        y_pred = ensemble.predict(self.X_test_processed)
        y_pred_proba = ensemble.predict_proba(self.X_test_processed) if hasattr(ensemble, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Store results
        results = {
            'model': ensemble,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'metrics': metrics,
            'component_models': models_to_ensemble
        }
        
        self.models['ensemble'] = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive model metrics."""
        metrics = {}
        
        if self.problem_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            if y_pred_proba is not None:
                if y_pred_proba.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        
        return metrics
    
    def feature_selection(self, method: str = 'mutual_info', 
                         n_features: int = 20) -> Dict:
        """
        Perform feature selection.
        
        Args:
            method: Feature selection method
            n_features: Number of features to select
            
        Returns:
            Dictionary with selected features and results
        """
        print(f"🔍 Performing Feature Selection using {method}...")
        
        if method == 'mutual_info':
            if self.problem_type == 'regression':
                selector = SelectKBest(score_func=f_regression, k=n_features)
            else:
                selector = SelectKBest(score_func=f_classif, k=n_features)
        
        elif method == 'recursive':
            if self.problem_type == 'regression':
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        
        elif method == 'tree_based':
            if self.problem_type == 'regression':
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, max_features=n_features)
        
        # Fit selector
        X_selected = selector.fit_transform(self.X_train_processed, self.y_train)
        selected_features = self.X_train_processed.columns[selector.get_support()].tolist()
        
        # Train model with selected features
        if self.problem_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_selected, self.y_train)
        y_pred = model.predict(selector.transform(self.X_test_processed))
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        results = {
            'selector': selector,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'metrics': metrics,
            'feature_scores': selector.scores_ if hasattr(selector, 'scores_') else None
        }
        
        self.results['feature_selection'] = results
        return results
    
    def model_interpretation(self, model_name: str = 'random_forest') -> Dict:
        """
        Perform model interpretation using SHAP and LIME.
        
        Args:
            model_name: Name of the model to interpret
            
        Returns:
            Dictionary with interpretation results
        """
        print(f"🔍 Interpreting {model_name} model...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]['model']
        
        # SHAP analysis
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict_proba, self.X_train_processed[:100])
            shap_values = explainer.shap_values(self.X_test_processed[:100])
        else:
            explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, self.X_train_processed[:100])
            shap_values = explainer.shap_values(self.X_test_processed[:100])
        
        # LIME analysis for a sample instance
        lime_explainer = LimeTabularExplainer(
            self.X_train_processed.values,
            feature_names=self.X_train_processed.columns,
            class_names=['Class 0', 'Class 1'] if self.problem_type == 'classification' else None,
            mode='classification' if self.problem_type == 'classification' else 'regression'
        )
        
        sample_idx = 0
        lime_exp = lime_explainer.explain_instance(
            self.X_test_processed.iloc[sample_idx].values,
            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
            num_features=10
        )
        
        results = {
            'shap_values': shap_values,
            'shap_explainer': explainer,
            'lime_explanation': lime_exp,
            'sample_instance': self.X_test_processed.iloc[sample_idx]
        }
        
        self.results['interpretation'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive model performance report."""
        report = []
        report.append("=" * 60)
        report.append("🤖 ADVANCED PREDICTIVE MODELING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset information
        report.append("📊 DATASET INFORMATION")
        report.append("-" * 30)
        report.append(f"Problem type: {self.problem_type}")
        report.append(f"Training samples: {len(self.X_train):,}")
        report.append(f"Testing samples: {len(self.X_test):,}")
        report.append(f"Features: {len(self.X_train_processed.columns)}")
        report.append(f"Categorical features: {len(self.categorical_features)}")
        report.append(f"Numerical features: {len(self.numerical_features)}")
        report.append("")
        
        # Model performance comparison
        if self.models:
            report.append("🏆 MODEL PERFORMANCE COMPARISON")
            report.append("-" * 40)
            
            # Create comparison table
            comparison_data = []
            for model_name, result in self.models.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    if self.problem_type == 'regression':
                        comparison_data.append({
                            'Model': model_name,
                            'RMSE': f"{metrics.get('rmse', 0):.4f}",
                            'MAE': f"{metrics.get('mae', 0):.4f}",
                            'R²': f"{metrics.get('r2', 0):.4f}",
                            'MAPE': f"{metrics.get('mape', 0):.2f}%"
                        })
                    else:
                        comparison_data.append({
                            'Model': model_name,
                            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                            'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}"
                        })
            
            comparison_df = pd.DataFrame(comparison_data)
            report.append(comparison_df.to_string(index=False))
            report.append("")
        
        # Feature importance (for tree-based models)
        if 'random_forest' in self.models and 'feature_importance' in self.models['random_forest']:
            report.append("🎯 TOP 10 FEATURE IMPORTANCE (Random Forest)")
            report.append("-" * 45)
            importance_df = self.models['random_forest']['feature_importance'].head(10)
            for _, row in importance_df.iterrows():
                report.append(f"{row['feature']}: {row['importance']:.4f}")
            report.append("")
        
        # Feature selection results
        if 'feature_selection' in self.results:
            fs = self.results['feature_selection']
            report.append("🔍 FEATURE SELECTION RESULTS")
            report.append("-" * 30)
            report.append(f"Selected features: {fs['n_selected']}")
            report.append(f"Performance with selected features:")
            for metric, value in fs['metrics'].items():
                report.append(f"  {metric}: {value:.4f}")
            report.append("")
        
        return "\n".join(report)
    
    def plot_results(self, save_path: str = None):
        """Create comprehensive visualization of results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'Feature Importance',
                          'Prediction vs Actual', 'Residuals Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Model performance comparison
        if self.models:
            model_names = list(self.models.keys())
            if self.problem_type == 'regression':
                metric = 'rmse'
            else:
                metric = 'accuracy'
            
            metric_values = [self.models[name]['metrics'].get(metric, 0) for name in model_names]
            
            fig.add_trace(
                go.Bar(x=model_names, y=metric_values, name=f'{metric.upper()}'),
                row=1, col=1
            )
        
        # Feature importance
        if 'random_forest' in self.models and 'feature_importance' in self.models['random_forest']:
            importance_df = self.models['random_forest']['feature_importance'].head(10)
            fig.add_trace(
                go.Bar(x=importance_df['feature'], y=importance_df['importance'], 
                      name='Feature Importance'),
                row=1, col=2
            )
        
        # Prediction vs Actual (using best model)
        if self.models:
            best_model_name = min(self.models.keys(), 
                                key=lambda x: self.models[x]['metrics'].get('rmse' if self.problem_type == 'regression' else 'accuracy', float('inf')))
            y_pred = self.models[best_model_name]['predictions']
            
            fig.add_trace(
                go.Scatter(x=self.y_test, y=y_pred, mode='markers', 
                          name=f'{best_model_name} Predictions'),
                row=2, col=1
            )
            
            # Add diagonal line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
                row=2, col=1
            )
        
        # Residuals plot
        if self.problem_type == 'regression' and self.models:
            residuals = self.y_test - y_pred
            fig.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
                row=2, col=2
            )
            
            # Add horizontal line at y=0
            fig.add_trace(
                go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0], 
                          mode='lines', name='Zero Line', line=dict(dash='dash')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Predictive Modeling Results")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def run_predictive_modeling(data_path: str = "data/simulated_customer_data.csv",
                          target_col: str = 'customer_lifetime_value',
                          problem_type: str = 'regression') -> AdvancedPredictiveModeler:
    """
    Run comprehensive predictive modeling pipeline.
    
    Args:
        data_path: Path to the data
        target_col: Target variable column
        problem_type: 'regression' or 'classification'
        
    Returns:
        AdvancedPredictiveModeler instance with results
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize modeler
    modeler = AdvancedPredictiveModeler(
        data=data,
        target_col=target_col,
        problem_type=problem_type
    )
    
    # Train ensemble models
    modeler.train_ensemble_models()
    
    # Train deep learning model
    modeler.train_deep_learning_model()
    
    # Create ensemble
    modeler.create_ensemble(['random_forest', 'xgboost', 'lightgbm'])
    
    # Feature selection
    modeler.feature_selection()
    
    # Model interpretation
    modeler.model_interpretation('random_forest')
    
    # Generate report
    report = modeler.generate_report()
    print(report)
    
    return modeler


if __name__ == "__main__":
    # Run predictive modeling
    modeler = run_predictive_modeling()
    
    # Create visualizations
    modeler.plot_results("predictive_modeling_results.html")
