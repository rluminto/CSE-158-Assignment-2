"""
Data loading and cleaning utilities for the Food.com recipe recommender project.
"""

import pandas as pd
import numpy as np
import ast
from typing import Tuple, List


def load_raw_data(recipes_path: str = "../datasets/RAW_recipes.csv", 
                  interactions_path: str = "../datasets/RAW_interactions.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw recipes and interactions data.
    
    Args:
        recipes_path: Path to RAW_recipes.csv
        interactions_path: Path to RAW_interactions.csv
        
    Returns:
        Tuple of (recipes_df, interactions_df)
    """
    recipes = pd.read_csv(recipes_path)
    interactions = pd.read_csv(interactions_path)
    
    return recipes, interactions


def parse_nutrition(nutrition_str: str) -> List[float]:
    """
    Parse nutrition string into list of floats.
    
    Args:
        nutrition_str: String representation of nutrition list
        
    Returns:
        List of 7 floats: [calories, total_fat_pdv, sugar_pdv, sodium_pdv, 
                          protein_pdv, saturated_fat_pdv, carbs_pdv]
    """
    try:
        return ast.literal_eval(nutrition_str)
    except:
        return [np.nan] * 7


def extract_nutrition_features(recipes: pd.DataFrame) -> pd.DataFrame:
    """
    Extract nutrition features from the nutrition column.
    
    Args:
        recipes: DataFrame with 'nutrition' column
        
    Returns:
        DataFrame with extracted nutrition columns added
    """
    recipes = recipes.copy()
    
    # Parse nutrition into separate columns
    nutrition_cols = ['calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 
                     'protein_pdv', 'saturated_fat_pdv', 'carbs_pdv']
    
    nutrition_data = recipes['nutrition'].apply(parse_nutrition)
    nutrition_df = pd.DataFrame(nutrition_data.tolist(), columns=nutrition_cols, index=recipes.index)
    
    # Add to recipes dataframe
    for col in nutrition_cols:
        recipes[col] = nutrition_df[col]
    
    return recipes


def define_healthiness(recipes: pd.DataFrame, 
                      calorie_threshold: float = 500,
                      sugar_threshold: float = 30,
                      satfat_threshold: float = 30) -> pd.DataFrame:
    """
    Define healthiness indicators based on nutrition thresholds.
    
    Args:
        recipes: DataFrame with nutrition columns
        calorie_threshold: Maximum calories for "healthy"
        sugar_threshold: Maximum sugar_pdv for "healthy"
        satfat_threshold: Maximum saturated_fat_pdv for "healthy"
        
    Returns:
        DataFrame with healthiness columns added
    """
    recipes = recipes.copy()
    
    # Individual flags
    recipes['is_low_cal'] = (recipes['calories'] <= calorie_threshold).astype(int)
    recipes['is_low_sugar'] = (recipes['sugar_pdv'] <= sugar_threshold).astype(int)
    recipes['is_low_satfat'] = (recipes['saturated_fat_pdv'] <= satfat_threshold).astype(int)
    
    # Combined healthiness indicator
    recipes['is_healthy'] = ((recipes['is_low_cal'] == 1) & 
                            (recipes['is_low_sugar'] == 1) & 
                            (recipes['is_low_satfat'] == 1)).astype(int)
    
    # Composite health score (lower is healthier)
    # Normalize calories to similar scale as pdv values
    calories_normalized = recipes['calories'] / 100.0
    recipes['health_score'] = (0.4 * calories_normalized + 
                              0.3 * recipes['sugar_pdv'] + 
                              0.3 * recipes['saturated_fat_pdv'])
    
    return recipes


def clean_interactions(interactions: pd.DataFrame, drop_zero_ratings: bool = True) -> pd.DataFrame:
    """
    Clean interactions data.
    
    Args:
        interactions: Raw interactions DataFrame
        drop_zero_ratings: Whether to drop rows with rating=0
        
    Returns:
        Cleaned interactions DataFrame
    """
    interactions = interactions.copy()
    
    # Drop missing values in key columns
    interactions = interactions.dropna(subset=['user_id', 'recipe_id', 'rating'])
    
    # Convert date to datetime
    interactions['date'] = pd.to_datetime(interactions['date'])
    
    # Handle zero ratings (often means no rating was given)
    if drop_zero_ratings:
        interactions = interactions[interactions['rating'] > 0]
    
    return interactions


def clean_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """
    Clean recipes data and add derived features.
    
    Args:
        recipes: Raw recipes DataFrame
        
    Returns:
        Cleaned recipes DataFrame
    """
    recipes = recipes.copy()
    
    # Convert submitted date
    recipes['submitted'] = pd.to_datetime(recipes['submitted'])
    
    # Extract nutrition features
    recipes = extract_nutrition_features(recipes)
    
    # Define healthiness indicators
    recipes = define_healthiness(recipes)
    
    # Cap extremely high minutes values (outliers)
    recipes['minutes_capped'] = recipes['minutes'].clip(upper=recipes['minutes'].quantile(0.99))
    
    # Log transform for skewed features
    recipes['log_minutes'] = np.log1p(recipes['minutes_capped'])
    recipes['log_n_steps'] = np.log1p(recipes['n_steps'])
    recipes['log_n_ingredients'] = np.log1p(recipes['n_ingredients'])
    
    return recipes


def filter_sparse_users_recipes(interactions: pd.DataFrame, 
                                min_user_interactions: int = 5,
                                min_recipe_interactions: int = 5) -> pd.DataFrame:
    """
    Filter out users and recipes with too few interactions.
    
    Args:
        interactions: Interactions DataFrame
        min_user_interactions: Minimum number of interactions per user
        min_recipe_interactions: Minimum number of interactions per recipe
        
    Returns:
        Filtered interactions DataFrame
    """
    interactions = interactions.copy()
    
    # Iteratively filter until stable
    prev_len = 0
    while len(interactions) != prev_len:
        prev_len = len(interactions)
        
        # Filter users
        user_counts = interactions['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        interactions = interactions[interactions['user_id'].isin(valid_users)]
        
        # Filter recipes
        recipe_counts = interactions['recipe_id'].value_counts()
        valid_recipes = recipe_counts[recipe_counts >= min_recipe_interactions].index
        interactions = interactions[interactions['recipe_id'].isin(valid_recipes)]
    
    return interactions


def create_binary_target(interactions: pd.DataFrame, rating_threshold: int = 4) -> pd.DataFrame:
    """
    Create binary 'like' target variable.
    
    Args:
        interactions: Interactions DataFrame with 'rating' column
        rating_threshold: Minimum rating to be considered a 'like'
        
    Returns:
        DataFrame with 'is_like' column added
    """
    interactions = interactions.copy()
    interactions['is_like'] = (interactions['rating'] >= rating_threshold).astype(int)
    return interactions


def get_data_summary(recipes: pd.DataFrame, interactions: pd.DataFrame) -> dict:
    """
    Get summary statistics about the dataset.
    
    Args:
        recipes: Recipes DataFrame
        interactions: Interactions DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        'n_recipes': len(recipes),
        'n_unique_recipes_in_interactions': interactions['recipe_id'].nunique(),
        'n_users': interactions['user_id'].nunique(),
        'n_interactions': len(interactions),
        'avg_rating': interactions['rating'].mean(),
        'rating_distribution': interactions['rating'].value_counts().sort_index().to_dict(),
        'healthy_recipe_fraction': recipes['is_healthy'].mean() if 'is_healthy' in recipes.columns else None,
    }
    
    return summary
