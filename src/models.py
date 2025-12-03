"""
Model implementations for the Food.com recipe recommender project.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self):
        self.prediction = None
    
    def fit(self, y_train):
        """Fit the baseline model."""
        raise NotImplementedError
    
    def predict_proba(self, X):
        """Predict probabilities."""
        n_samples = len(X) if hasattr(X, '__len__') else 1
        probs = np.full((n_samples, 2), [1 - self.prediction, self.prediction])
        return probs
    
    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GlobalAverageBaseline(BaselineModel):
    """Baseline that predicts the global average like rate."""
    
    def fit(self, y_train):
        """Compute global average from training data."""
        self.prediction = y_train.mean()
        return self
    
    def __repr__(self):
        return f"GlobalAverageBaseline(prediction={self.prediction:.4f})"


class RecipeAverageBaseline(BaselineModel):
    """Baseline that predicts based on recipe-specific like rates."""
    
    def __init__(self):
        super().__init__()
        self.recipe_rates = {}
        self.global_rate = None
    
    def fit(self, train_df, recipe_col='recipe_id', target_col='is_like'):
        """Compute recipe-specific like rates."""
        self.recipe_rates = train_df.groupby(recipe_col)[target_col].mean().to_dict()
        self.global_rate = train_df[target_col].mean()
        return self
    
    def predict_proba(self, test_df, recipe_col='recipe_id'):
        """Predict using recipe rates, fall back to global for unseen recipes."""
        predictions = test_df[recipe_col].map(self.recipe_rates).fillna(self.global_rate).values
        probs = np.column_stack([1 - predictions, predictions])
        return probs
    
    def __repr__(self):
        return f"RecipeAverageBaseline(n_recipes={len(self.recipe_rates)}, global_rate={self.global_rate:.4f})"


class UserAverageBaseline(BaselineModel):
    """Baseline that predicts based on user-specific like rates."""
    
    def __init__(self):
        super().__init__()
        self.user_rates = {}
        self.global_rate = None
    
    def fit(self, train_df, user_col='user_id', target_col='is_like'):
        """Compute user-specific like rates."""
        self.user_rates = train_df.groupby(user_col)[target_col].mean().to_dict()
        self.global_rate = train_df[target_col].mean()
        return self
    
    def predict_proba(self, test_df, user_col='user_id'):
        """Predict using user rates, fall back to global for unseen users."""
        predictions = test_df[user_col].map(self.user_rates).fillna(self.global_rate).values
        probs = np.column_stack([1 - predictions, predictions])
        return probs
    
    def __repr__(self):
        return f"UserAverageBaseline(n_users={len(self.user_rates)}, global_rate={self.global_rate:.4f})"


class LogisticRegressionModel:
    """Logistic regression model with preprocessing pipeline."""
    
    def __init__(self, C=1.0, class_weight=None, random_state=42):
        """
        Initialize logistic regression model.
        
        Args:
            C: Inverse of regularization strength
            class_weight: Weights for classes (None, 'balanced', or dict)
            random_state: Random seed
        """
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=C,
                class_weight=class_weight,
                random_state=random_state,
                max_iter=1000
            ))
        ])
        
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X_train, y_train, feature_names=None):
        """
        Fit the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            feature_names: Optional list of feature names
        """
        self.pipeline.fit(X_train, y_train)
        self.feature_names = feature_names
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.pipeline.predict_proba(X)
    
    def get_coefficients(self):
        """Get model coefficients with feature names if available."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        coefs = self.pipeline.named_steps['classifier'].coef_[0]
        
        if self.feature_names is not None:
            return pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefs
            }).sort_values('coefficient', key=abs, ascending=False)
        else:
            return coefs
    
    def get_intercept(self):
        """Get model intercept."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting intercept")
        return self.pipeline.named_steps['classifier'].intercept_[0]
    
    def __repr__(self):
        return f"LogisticRegressionModel(C={self.C}, class_weight={self.class_weight})"


def tune_hyperparameters(X_train, y_train, X_val, y_val, 
                        C_values=[0.01, 0.1, 1, 10],
                        metric='roc_auc',
                        feature_names=None):
    """
    Simple hyperparameter tuning for logistic regression.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        C_values: List of C values to try
        metric: Metric to optimize ('roc_auc' or 'log_loss')
        feature_names: Optional list of feature names
        
    Returns:
        Tuple of (best_model, results_df)
    """
    from sklearn.metrics import roc_auc_score, log_loss
    
    results = []
    best_score = -np.inf if metric == 'roc_auc' else np.inf
    best_model = None
    
    for C in C_values:
        model = LogisticRegressionModel(C=C)
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Evaluate on validation set
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        if metric == 'roc_auc':
            score = roc_auc_score(y_val, y_val_proba)
            is_better = score > best_score
        else:  # log_loss
            score = log_loss(y_val, y_val_proba)
            is_better = score < best_score
        
        results.append({
            'C': C,
            'roc_auc': roc_auc_score(y_val, y_val_proba),
            'log_loss': log_loss(y_val, y_val_proba)
        })
        
        if is_better:
            best_score = score
            best_model = model
    
    results_df = pd.DataFrame(results)
    
    return best_model, results_df


class RecipeRecommender:
    """Simple recipe recommender based on predicted like probabilities."""
    
    def __init__(self, model, feature_cols):
        """
        Initialize recommender.
        
        Args:
            model: Fitted model with predict_proba method
            feature_cols: List of feature column names
        """
        self.model = model
        self.feature_cols = feature_cols
    
    def recommend(self, user_id, candidate_df, top_k=10, 
                 health_weight=0.0, health_penalty_col='health_score'):
        """
        Recommend top-K recipes for a user.
        
        Args:
            user_id: User ID to recommend for
            candidate_df: DataFrame of candidate (user, recipe) pairs with features
            top_k: Number of recommendations to return
            health_weight: Weight for health penalty (0 = no adjustment)
            health_penalty_col: Column to use for health penalty
            
        Returns:
            DataFrame with top-K recommendations and their scores
        """
        # Filter to candidates for this user
        user_candidates = candidate_df[candidate_df['user_id'] == user_id].copy()
        
        if len(user_candidates) == 0:
            return pd.DataFrame()
        
        # Get available features
        available_features = [col for col in self.feature_cols if col in user_candidates.columns]
        
        # Predict probabilities
        X = user_candidates[available_features].values
        probs = self.model.predict_proba(X)[:, 1]
        user_candidates['pred_like_prob'] = probs
        
        # Apply health adjustment if requested
        if health_weight > 0 and health_penalty_col in user_candidates.columns:
            # Normalize health penalty to [0, 1] range
            health_vals = user_candidates[health_penalty_col].values
            health_normalized = (health_vals - health_vals.min()) / (health_vals.max() - health_vals.min() + 1e-8)
            user_candidates['score'] = probs - health_weight * health_normalized
        else:
            user_candidates['score'] = probs
        
        # Sort and get top-K
        recommendations = user_candidates.nlargest(top_k, 'score')
        
        return recommendations[['recipe_id', 'score', 'pred_like_prob'] + 
                              ([health_penalty_col] if health_penalty_col in recommendations.columns else [])]
    
    def recommend_batch(self, user_ids, candidate_df, top_k=10,
                       health_weight=0.0, health_penalty_col='health_score'):
        """
        Recommend for multiple users.
        
        Args:
            user_ids: List of user IDs
            candidate_df: DataFrame of all candidates
            top_k: Number of recommendations per user
            health_weight: Weight for health penalty
            health_penalty_col: Column to use for health penalty
            
        Returns:
            Dictionary mapping user_id to recommendations DataFrame
        """
        recommendations = {}
        for user_id in user_ids:
            recs = self.recommend(user_id, candidate_df, top_k, 
                                health_weight, health_penalty_col)
            if len(recs) > 0:
                recommendations[user_id] = recs
        
        return recommendations
