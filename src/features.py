"""
Feature engineering utilities for the Food.com recipe recommender project.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_user_features(interactions: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-level aggregate features.
    
    Args:
        interactions: Interactions DataFrame
        recipes: Recipes DataFrame with nutrition info
        
    Returns:
        DataFrame with one row per user and their aggregate features
    """
    # Merge to get recipe features for each interaction
    df = interactions.merge(recipes[['id', 'calories', 'minutes']], 
                           left_on='recipe_id', right_on='id', how='left')
    
    user_features = df.groupby('user_id').agg({
        'rating': ['mean', 'count', 'std'],
        'calories': 'mean',
        'minutes': 'mean',
        'recipe_id': 'nunique'
    }).reset_index()
    
    # Flatten column names
    user_features.columns = ['user_id', 'user_mean_rating', 'user_num_ratings', 
                            'user_rating_std', 'user_mean_calories', 
                            'user_mean_minutes', 'user_num_unique_recipes']
    
    # Log transform count features
    user_features['user_log_num_ratings'] = np.log1p(user_features['user_num_ratings'])
    
    # Fill NaN in std with 0 (users with only 1 rating)
    user_features['user_rating_std'] = user_features['user_rating_std'].fillna(0)
    
    return user_features


def compute_recipe_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute recipe-level aggregate features from interactions.
    
    Args:
        interactions: Interactions DataFrame
        
    Returns:
        DataFrame with one row per recipe and their aggregate features
    """
    recipe_features = interactions.groupby('recipe_id').agg({
        'rating': ['mean', 'count', 'std'],
        'user_id': 'nunique'
    }).reset_index()
    
    # Flatten column names
    recipe_features.columns = ['recipe_id', 'recipe_mean_rating', 'recipe_num_ratings',
                              'recipe_rating_std', 'recipe_num_unique_users']
    
    # Log transform count features
    recipe_features['recipe_log_num_ratings'] = np.log1p(recipe_features['recipe_num_ratings'])
    
    # Fill NaN in std with 0
    recipe_features['recipe_rating_std'] = recipe_features['recipe_rating_std'].fillna(0)
    
    return recipe_features


def create_modeling_dataset(interactions: pd.DataFrame, 
                           recipes: pd.DataFrame,
                           user_features: pd.DataFrame = None,
                           recipe_features: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create the full modeling dataset by joining interactions with all features.
    
    Args:
        interactions: Interactions DataFrame (should have 'is_like' target)
        recipes: Recipes DataFrame with all features
        user_features: Optional precomputed user features
        recipe_features: Optional precomputed recipe features
        
    Returns:
        Complete modeling DataFrame
    """
    # Compute features if not provided
    if user_features is None:
        user_features = compute_user_features(interactions, recipes)
    if recipe_features is None:
        recipe_features = compute_recipe_features(interactions)
    
    # Start with interactions
    df = interactions.copy()
    
    # Merge recipe features (from recipes table)
    recipe_cols = ['id', 'minutes', 'n_steps', 'n_ingredients',
                  'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv',
                  'protein_pdv', 'saturated_fat_pdv', 'carbs_pdv',
                  'is_healthy', 'is_low_cal', 'is_low_sugar', 'is_low_satfat',
                  'health_score', 'log_minutes', 'log_n_steps', 'log_n_ingredients',
                  'submitted']
    
    available_cols = ['id'] + [col for col in recipe_cols if col in recipes.columns and col != 'id']
    df = df.merge(recipes[available_cols], left_on='recipe_id', right_on='id', how='left')
    
    # Merge recipe aggregate features (from interactions)
    df = df.merge(recipe_features, on='recipe_id', how='left')
    
    # Merge user features
    df = df.merge(user_features, on='user_id', how='left')
    
    # Add interaction-specific features
    if 'date' in df.columns and 'submitted' in df.columns:
        df['days_since_submission'] = (df['date'] - df['submitted']).dt.days
        df['log_days_since_submission'] = np.log1p(df['days_since_submission'].clip(lower=0))
    
    if 'date' in df.columns:
        df['interaction_year'] = df['date'].dt.year
        df['interaction_month'] = df['date'].dt.month
        df['interaction_dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['interaction_dayofweek'] >= 5).astype(int)
    
    return df


def get_feature_columns(for_modeling: bool = True) -> list:
    """
    Get list of feature columns to use for modeling.
    
    Args:
        for_modeling: If True, return features for model input. 
                     If False, return all feature columns.
    
    Returns:
        List of feature column names
    """
    # User features
    user_cols = [
        'user_mean_rating',
        'user_log_num_ratings',
        'user_rating_std',
        'user_mean_calories',
        'user_mean_minutes'
    ]
    
    # Recipe metadata features
    recipe_meta_cols = [
        'log_minutes',
        'log_n_steps',
        'log_n_ingredients'
    ]
    
    # Recipe nutrition features
    nutrition_cols = [
        'calories',
        'total_fat_pdv',
        'sugar_pdv',
        'sodium_pdv',
        'protein_pdv',
        'saturated_fat_pdv',
        'carbs_pdv',
        'is_healthy',
        'health_score'
    ]
    
    # Recipe aggregate features
    recipe_agg_cols = [
        'recipe_mean_rating',
        'recipe_log_num_ratings',
        'recipe_rating_std'
    ]
    
    # Interaction features
    interaction_cols = [
        'log_days_since_submission',
        'is_weekend'
    ]
    
    if for_modeling:
        # Return commonly available features for modeling
        return (user_cols + recipe_meta_cols + nutrition_cols + 
                recipe_agg_cols + interaction_cols)
    else:
        # Return all feature columns
        return (user_cols + recipe_meta_cols + nutrition_cols + 
                recipe_agg_cols + interaction_cols)


def split_temporal_per_user(df: pd.DataFrame, 
                           train_ratio: float = 0.6,
                           val_ratio: float = 0.2,
                           test_ratio: float = 0.2,
                           date_col: str = 'date',
                           user_col: str = 'user_id') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally per user to avoid data leakage.
    
    For each user, their interactions are sorted by date and split into
    train/val/test according to the specified ratios.
    
    Args:
        df: DataFrame with interactions
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        date_col: Name of date column
        user_col: Name of user column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    df = df.copy()
    df = df.sort_values([user_col, date_col])
    
    # Assign split based on position within each user's history
    def assign_split(group):
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        split = ['train'] * train_end + ['val'] * (val_end - train_end) + ['test'] * (n - val_end)
        group['split'] = split
        return group
    
    df = df.groupby(user_col, group_keys=False).apply(assign_split)
    
    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    val_df = df[df['split'] == 'val'].drop(columns=['split'])
    test_df = df[df['split'] == 'test'].drop(columns=['split'])
    
    return train_df, val_df, test_df


def prepare_features_for_training(train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  feature_cols: list,
                                  target_col: str = 'is_like') -> Tuple:
    """
    Prepare feature matrices and target vectors for training.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        target_col: Name of target column
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, available_features)
    """
    # Filter to only available features
    available_features = [col for col in feature_cols if col in train_df.columns]
    
    # Drop rows with missing values in features or target
    train_clean = train_df[available_features + [target_col]].dropna()
    val_clean = val_df[available_features + [target_col]].dropna()
    test_clean = test_df[available_features + [target_col]].dropna()
    
    # Extract features and targets
    X_train = train_clean[available_features].values
    y_train = train_clean[target_col].values
    
    X_val = val_clean[available_features].values
    y_val = val_clean[target_col].values
    
    X_test = test_clean[available_features].values
    y_test = test_clean[target_col].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, available_features
