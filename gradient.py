import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

# Define a function for hyperparameter tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

# Load the dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Data preprocessing steps (as in the original script)
# Create a copy of the data to avoid SettingWithCopyWarning
data_cleaned = data.copy()

# Remove rows where all elements are NaN
data_cleaned.dropna(how='all', inplace=True)

# Check for missing values in each column
missing_values = data_cleaned.isnull().sum()
#print(missing_values)

# Impute missing values in 'avclass' with a new category 'unknown'
data_cleaned.loc[:, 'avclass'].fillna('unknown', inplace=True)

# Add binary indicators for missing data
for column in ['optional.dll_characteristics', 'entry', 'imports', 'exports', 'datadirectories', 'sections']:
    data_cleaned.loc[:, f'{column}_missing'] = data_cleaned[column].isnull().astype(int)

# Extract features from 'sections' column and drop the original 'sections' column
data_cleaned.loc[:, 'num_sections'] = data_cleaned['sections'].str.count('|') + 1
data_cleaned.loc[:, 'section_desc_length'] = data_cleaned['sections'].apply(lambda x: len(str(x)))
data_cleaned.drop('sections', axis=1, inplace=True)

# Frequency encoding for 'avclass'
frequency = data_cleaned['avclass'].value_counts(normalize=True)  # Getting the frequency of each category
data_cleaned.loc[:, 'avclass'] = data_cleaned['avclass'].map(frequency)  # Mapping each category to its frequency

# Define the function for processing distribution-like columns
def process_distribution_column(column):
    # Split and convert to lists of numbers, then calculate statistics
    processed_column = column.str.split('|').apply(lambda x: [float(i) for i in x] if isinstance(x, list) else np.nan)
    stats = processed_column.apply(lambda x: pd.Series([np.mean(x), np.std(x)], index=['mean', 'std']) if x is not None else [np.nan, np.nan])
    return stats

# Process distribution-like columns and join them with the main dataframe
for col in ['histogram', 'byteentropy', 'printabledist']:
    stats = process_distribution_column(data_cleaned[col]).rename(columns=lambda x: f'{col}_{x}')
    data_cleaned = pd.concat([data_cleaned, stats], axis=1)
    data_cleaned.drop(col, axis=1, inplace=True)

# Convert 'appeared' to datetime and extract year and month
data_cleaned['appeared'] = pd.to_datetime(data_cleaned['appeared'])
data_cleaned['appeared_year'] = data_cleaned['appeared'].dt.year
data_cleaned['appeared_month'] = data_cleaned['appeared'].dt.month
data_cleaned.drop('appeared', axis=1, inplace=True)

# One-hot encode the nominal categorical columns
nominal_cols = ['coff.machine', 'optional.subsystem', 'optional.magic']
data_cleaned = pd.get_dummies(data_cleaned, columns=nominal_cols, prefix=nominal_cols)

# Encoding other categorical columns as needed
for col in ['entry', 'imports', 'exports', 'datadirectories', 
            'coff.characteristics', 'optional.dll_characteristics']:
    # Frequency encoding for high cardinality columns
    frequency = data_cleaned[col].value_counts(normalize=True)
    data_cleaned[col] = data_cleaned[col].map(frequency)

# Remove the 'md5' column and any 'unnamed' columns
cols_to_drop = ['md5'] + [col for col in data_cleaned.columns if 'unnamed' in col.lower()]
data_cleaned.drop(cols_to_drop, axis=1, inplace=True)

# Balancing the classes by undersampling the majority class
data_majority = data_cleaned[data_cleaned['label'] == 0]
data_minority = data_cleaned[data_cleaned['label'] == 1]
data_majority_downsampled = data_majority.sample(len(data_minority), random_state=42)
data_balanced = pd.concat([data_majority_downsampled, data_minority])


# Define features and target
X = data_balanced.drop('label', axis=1)
y = data_balanced['label']

# Imputation of missing values for numerical columns
imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply Variance Threshold to remove constant features
selector = VarianceThreshold()
X_var_threshold = selector.fit_transform(X_scaled)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_var_threshold, y, test_size=0.2695, random_state=42)

# Feature Selection: SelectKBest
select = SelectKBest(score_func=f_classif, k=2)  # Adjust 'k' as needed
X_train_selected = select.fit_transform(X_train, y_train)
X_val_selected = select.transform(X_val)

# Define the hyperparameters and their search ranges
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of trees in the ensemble
    'learning_rate': [0.01, 0.1, 0.2, 0.3],  # Learning rate for boosting
    'max_depth': [3, 4, 5, 6],  # Maximum depth of individual trees
}
# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',  # Use appropriate scoring metric
                           cv=5)  # Adjust the number of cross-validation folds

# Fit the grid search to your data

grid_search.fit(X_train_selected, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_gb = GradientBoostingClassifier(random_state=42, **best_params)
best_gb.fit(X_train_selected, y_train)

# Predict on the validation set
y_val_pred = best_gb.predict(X_val_selected)

# Evaluation
print("Model: GradientBoostingClassifier (Tuned)")
print(classification_report(y_val, y_val_pred))

# Cross-validation for the tuned model
scores = cross_val_score(best_gb, X_scaled, y, cv=5, scoring='f1_macro')
print(f"Cross-validated F1 scores: {scores}")

