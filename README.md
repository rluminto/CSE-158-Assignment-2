# Food.com Recipe Recommender System with Health Bias Analysis

## Project Overview

This project builds a recipe recommender system using the Food.com dataset to predict user preferences and analyze potential health bias in recommendations. The system addresses three key research questions:

1. **RQ1 (Prediction)**: How accurately can we predict whether a user will like a recipe (rating ≥ 4)?
2. **RQ2 (Health vs Ratings)**: Are healthier recipes systematically rated lower than less healthy recipes?
3. **RQ3 (Fairness)**: Does the recommender system exhibit bias toward unhealthy recipes?

## Repository Structure

```
.
├── datasets/                      # Data files
│   ├── RAW_recipes.csv           # Original recipe data
│   ├── RAW_interactions.csv      # Original interaction data
│   ├── recipes_clean.csv         # Cleaned recipe data (generated)
│   └── interactions_clean.csv    # Cleaned interaction data (generated)
│
├── src/                          # Source code modules
│   ├── data_utils.py            # Data loading and cleaning utilities
│   ├── features.py              # Feature engineering functions
│   ├── models.py                # Model implementations
│   └── eval_utils.py            # Evaluation metrics and visualizations
│
├── notebooks/                    # Jupyter notebooks (main analysis)
│   ├── 01_eda_data_understanding.ipynb
│   ├── 02_modeling_and_recommender.ipynb
│   └── 03_health_bias_and_fairness.ipynb
│
├── reports/                      # Generated reports and figures
│   ├── figures/                 # Saved visualizations
│   ├── model_evaluation.csv     # Model performance metrics
│   ├── feature_importance.csv   # Feature coefficients
│   └── health_tradeoff_results.csv
│
├── requirements.txt              # Python dependencies
├── implementation.txt            # Detailed implementation plan
├── instructions.txt             # Assignment instructions
└── README.md                    # This file
```

## Installation and Setup

### 1. Clone or Download the Repository

```powershell
cd "c:\Users\reine\Documents\UCSD Classes\Fall 25\CSE 158\CSE-158-Assignment-2"
```

### 2. Install Dependencies

Create a virtual environment (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install required packages:

```powershell
pip install -r requirements.txt
```

### 3. Verify Data Files

Ensure the following files are in the `datasets/` directory:
- `RAW_recipes.csv`
- `RAW_interactions.csv`

## Usage

### Running the Analysis

Execute the notebooks in order:

1. **01_eda_data_understanding.ipynb**
   - Loads and cleans raw data
   - Extracts nutrition features
   - Defines healthiness metrics
   - Generates exploratory visualizations
   - Saves cleaned data

2. **02_modeling_and_recommender.ipynb**
   - Creates binary prediction target (is_like)
   - Performs temporal train/val/test split
   - Trains baseline models and logistic regression
   - Builds recipe recommender system
   - Evaluates prediction and recommendation performance

3. **03_health_bias_and_fairness.ipynb**
   - Analyzes relationship between health and ratings
   - Measures bias in recommendations
   - Experiments with health-adjusted scoring
   - Visualizes quality-healthiness trade-offs

### Starting Jupyter

```powershell
jupyter lab
# or
jupyter notebook
```

Navigate to the `notebooks/` directory and run cells sequentially.

## Key Features

### Data Processing
- **Nutrition Parsing**: Extracts 7 nutrition metrics from stringified lists
- **Healthiness Definition**: Binary indicator based on calories (≤500), sugar (≤30% DV), and saturated fat (≤30% DV)
- **Sparse Filtering**: Removes users/recipes with fewer than 5 interactions
- **Temporal Splitting**: Per-user time-based train/val/test split (60/20/20)

### Models
- **Baselines**: Global average, recipe average, user average
- **Main Model**: Logistic regression with L2 regularization
- **Features**: User aggregates, recipe metadata, nutrition metrics, interaction context

### Evaluation
- **Classification**: AUC-ROC, accuracy, precision, recall, F1, log-loss, Brier score
- **Recommendation**: Precision@K, Recall@K (K=5, 10)
- **Health Bias**: Fraction of healthy recipes in recommendations vs candidate pool

### Visualizations
All figures are saved to `reports/figures/`:
- Rating distribution
- Nutrition vs rating relationships
- Healthy vs unhealthy comparisons
- ROC curves
- Feature importance
- Health bias metrics
- Quality-healthiness trade-off curves

## Research Questions and Findings

### RQ1: Prediction Accuracy
The logistic regression model achieves strong performance in predicting whether users will like recipes, significantly outperforming baseline approaches. Key predictive features include historical user ratings, recipe popularity, and nutrition metrics.

### RQ2: Health vs Ratings
Statistical analysis reveals whether healthier recipes receive systematically different ratings than less healthy alternatives, controlling for other factors like cooking time and complexity.

### RQ3: Recommender Bias
The baseline recommender (pure prediction-based) exhibits measurable bias toward less healthy recipes. By adjusting the scoring function with a health penalty parameter (α), we can increase healthy recommendations with controlled quality trade-offs.

## Customization

### Adjusting Healthiness Criteria

Edit `src/data_utils.py`, function `define_healthiness()`:

```python
def define_healthiness(recipes, 
                      calorie_threshold=500,      
                      sugar_threshold=30,        
                      satfat_threshold=30):      
```

### Modifying Features

Edit `src/features.py`, function `get_feature_columns()` to add/remove features.

### Tuning Health Weight

In notebook 03, adjust the `health_weights` list to experiment with different α values:

```python
health_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
```

## Dependencies

Core packages (see `requirements.txt`):
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## Limitations and Future Work

### Current Limitations
1. **Healthiness Definition**: Simple threshold-based approach; doesn't capture full nutritional complexity
2. **Cold Start**: Limited handling of new users/recipes with no history
3. **Text Features**: Currently unused (ingredients, descriptions, steps)
4. **Scalability**: In-memory processing limits dataset size

### Future Enhancements
1. **Advanced Models**: Neural networks, matrix factorization, transformers
2. **Text Embeddings**: TF-IDF or BERT-based features from recipe descriptions
3. **Personalized Health**: User-specific health preferences and dietary restrictions
4. **Temporal Dynamics**: Model changes in user preferences over time
5. **Diversity**: Ensure recommendation diversity beyond accuracy

## Authors and Acknowledgments

- **Course**: CSE 158 - Web Mining and Recommender Systems
- **Institution**: UC San Diego
- **Dataset**: Food.com Recipes and Interactions (Kaggle)

## License

This project is for educational purposes as part of CSE 158 at UCSD.

## Citation
Generating Personalized Recipes from Historical User Preferences
Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley
EMNLP, 2019

SHARE: a System for Hierarchical Assistive Recipe Editing
Shuyang Li, Yufei Li, Jianmo Ni, Julian McAuley
EMNLP, 2022

## Author
Authors: 
- Reiner Luminto (A18554372)
- Stephanie Patricia Anshell (A18503005)
- Bryan Valerian Lie (A18545672) 

Course: CSE 158 — Fall 2025

Instructor: Julian McAuley

---

**Note**: This README provides a comprehensive guide to the project structure and usage. For detailed implementation steps and methodology, refer to `implementation.txt`.
