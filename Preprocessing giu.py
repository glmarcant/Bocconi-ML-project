#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer, make_column_selector
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model  import LinearRegression
from sklearn.metrics       import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# 1) Load and drop rows with missing target
df = pd.read_csv('data/train.csv')
df = df.dropna(subset=['value_eur'])


# 2) Split into X / y
y = df.pop('value_eur')
X = df


# 3) Preprocessing

# MISSING VALUES
# Identify features with the most missing values
missing_counts = X.isnull().sum().sort_values(ascending=False)
print("Missing values per feature:")
print(missing_counts)

# Drop the six features with the most missing values
features_to_drop = missing_counts.head(6).index
X = X.drop(columns=features_to_drop)

# IRRELEVANT FEATURES
# Drop features that are not relevant for the model
irrelevant_features = [
    'Unnamed: 0', 'id', 'short_name', 'long_name', 'dob', 'nationality_id', 'club_jersey_number', 'club_team_id'
]

# Drop the manually specified features
X = X.drop(columns=irrelevant_features)

# MODIFY FEATURES THAT ARE WRITTEN IN FORM OF DATES
# Convert 'club_joined' to datetime and extract year
X['club_joined'] = pd.to_datetime(X['club_joined'], errors='coerce')
X['club_joined_year'] = X['club_joined'].dt.year
X = X.drop(columns=['club_joined'])  # Drop the original column


# REMOVING VARIABLES BASED ON CORRELATION
# Temporarily add 'value_eur' back to X
X['value_eur'] = y

# Calculate the correlation matrix
correlation_matrix = X.corr(numeric_only=True)

# Find pairs of features with correlation greater than 0.7
high_correlation = correlation_matrix.abs() > 0.7
correlated_pairs = [
    (col1, col2)
    for col1 in correlation_matrix.columns
    for col2 in correlation_matrix.columns
    if col1 != col2 and high_correlation.loc[col1, col2]
]

# Get the correlation of each feature with 'value_eur'
value_eur_correlation = correlation_matrix['value_eur']

# Create a set to store the variables to remove
variables_to_remove = set()

# Compare correlations and add the variable with the lower correlation to 'value_eur' to the removal set
for col1, col2 in correlated_pairs:
    if value_eur_correlation[col1] > value_eur_correlation[col2]:
        variables_to_remove.add(col2)
    else:
        variables_to_remove.add(col1)

# Remove 'value_eur' from X again
X = X.drop(columns=['value_eur'])

# Print the set of variables to remove
print("Variables to remove based on correlation with 'value_eur':")
print(variables_to_remove)
print(f"Number of variables to remove: {len(variables_to_remove)}")

# Remove the variables in variables_to_remove from X
X = X.drop(columns=variables_to_remove, errors='ignore')


print(f"Number of features left: {X.shape[1]}")
print("Features left after removing correlated variables:")
print(X.columns.tolist())




#4) Feature Engineering


# TRANSFORMATION OF FEATURES
# Apply log transformation to the target variable since it is highly skewed
y = np.log1p(y)

# Apply squared transformation to the target variable to make it linear
X['age_squared'] = X['age'] ** 2


# Add an interaction term between 


# MODIFY FEATURES IN MACRO-CATEGORIES
# For visualizing all possible player positions
unique_positions = set()

# Loop through each entry and split by ", "
for pos in X['player_positions'].dropna():
    roles = [role.strip() for role in pos.split(',')]
    unique_positions.update(roles)

print(unique_positions)

# Mapping individual positions to macro-categories
role_to_group = {
    'GK': 'Goalkeeper', 
    'CB': 'Defender', 'LB': 'Defender', 'RB': 'Defender', 'LWB': 'Defender', 'RWB': 'Defender',
    'CDM': 'Midfielder', 'CM': 'Midfielder', 'CAM': 'Midfielder', 'LM': 'Midfielder', 'RM': 'Midfielder',
    'ST': 'Forward', 'CF': 'Forward', 'LW': 'Forward', 'RW': 'Forward'
}

# Initialize binary columns
X['is_defender'] = 0
X['is_midfielder'] = 0
X['is_forward'] = 0
X['is_goalkeeper'] = 0

# Loop through player_positions and set dummies
for idx, pos in X['player_positions'].dropna().items():
    roles = [role.strip() for role in pos.split(',')]
    groups = set(role_to_group.get(role, 'Goalkeeper') for role in roles)
    
    if 'Defender' in groups:
        X.at[idx, 'is_defender'] = 1
    if 'Midfielder' in groups:
        X.at[idx, 'is_midfielder'] = 1
    if 'Forward' in groups:
        X.at[idx, 'is_forward'] = 1
    if 'Goalkeeper' in groups:
        X.at[idx, 'is_goalkeeper'] = 1

# Drop the original player_positions column
X = X.drop(columns=['player_positions'])

print(f"Number of features left after preprocessing: {X.shape[1]}")


# 5) Define the ColumnTransformer
preprocessor = ColumnTransformer([
    # numeric: median‐impute + scale
    ('num',
       Pipeline([
           ('imputer', SimpleImputer(strategy='median')),
           ('scaler',   StandardScaler())
       ]),
       make_column_selector(dtype_include=np.number)
    ),
    # categorical: constant‐impute + one‐hot
    ('cat',
       Pipeline([
           ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
           ('onehot',  OneHotEncoder(handle_unknown='ignore',
                                     sparse_output=False))
       ]),
       make_column_selector(dtype_exclude=np.number)
    )
],
    remainder='drop'  # drop anything else
)

# 6) Full pipeline with OLS baseline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model',      LinearRegression())
])

# 7) Define shuffled KFold cross-validator
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 8) Cross-validation with the shuffled splits
neg_mse_scores = cross_val_score(pipeline, X, y, scoring='neg_root_mean_squared_error', cv=cv)
r2_scores = cross_val_score(pipeline, X, y, scoring='r2', cv=cv)

# Convert negative RMSE scores to positive
rmse_scores = -neg_mse_scores

# 9) Report average and std
print(f"Cross-validated RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
print(f"Cross-validated R²:   {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
# %%