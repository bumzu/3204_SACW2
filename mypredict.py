import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold

def preprocess_data(data, is_training_data=True):
    # Create a copy of the data
    data_cleaned = data.copy()

    # Remove rows where all elements are NaN
    data_cleaned.dropna(how='all', inplace=True)

    # Check for missing values in each column
    missing_values = data_cleaned.isnull().sum()

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
    frequency = data_cleaned['avclass'].value_counts(normalize=True)
    data_cleaned.loc[:, 'avclass'] = data_cleaned['avclass'].map(frequency)

    # Define the function for processing distribution-like columns
    def process_distribution_column(column):
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

    # Encoding other categorical columns as needed
    for col in ['entry', 'imports', 'exports', 'datadirectories', 
                'coff.characteristics', 'optional.dll_characteristics']:
        # Frequency encoding for high cardinality columns
        frequency = data_cleaned[col].value_counts(normalize=True)
        data_cleaned[col] = data_cleaned[col].map(frequency)

    # Remove the 'md5' column and any 'unnamed' columns (which is the ID, which seems irrelevant)
    cols_to_drop = ['md5', 'Unnamed: 0']  # Add any columns you want to drop
    data_cleaned.drop(cols_to_drop, axis=1, inplace=True)

    # If it's training data and the label column exists, apply class balancing
    if is_training_data and 'label' in data:
        data_majority = data_cleaned[data_cleaned['label'] == 0]
        data_minority = data_cleaned[data_cleaned['label'] == 1]
        data_majority_downsampled = data_majority.sample(len(data_minority), random_state=42)
        data_cleaned = pd.concat([data_majority_downsampled, data_minority])

    # Split data into features and label if label column exists
    if 'label' in data_cleaned:
        features = data_cleaned.drop('label', axis=1)
        labels = data_cleaned['label']
        return features, labels
    else:
        return data_cleaned, None

# Check for correct amount of arguments
if len(sys.argv) != 3:
    print("Usage: python3 main.py <training_csv> <testing_csv>")
    print("Example: python3 main.py train.csv UnseenData.csv")
    sys.exit(1)

# Read file paths from command line arguments
train_file_path = sys.argv[1]
unseen_file_path = sys.argv[2]

# Check if the last two arguments are CSV files
if not train_file_path.endswith(".csv") or not unseen_file_path.endswith(".csv"):
    print("Error: Both arguments should be .csv files.")
    sys.exit(1)

# Load training data
try:
    train_data = pd.read_csv(train_file_path)
except FileNotFoundError:
    print(f"Error: File '{train_file_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading '{train_file_path}': {e}")
    sys.exit(1)

# Load unseen data
try:
    unseen_data = pd.read_csv(unseen_file_path)
except FileNotFoundError:
    print(f"Error: File '{unseen_file_path}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading '{unseen_file_path}': {e}")
    sys.exit(1)

# If no issues during load, preprocess training and unseen data
X_train, y_train = preprocess_data(train_data, is_training_data=True)
X_unseen, _ = preprocess_data(unseen_data, is_training_data=False)

# Combine, one-hot encode, and then split the data back
nominal_cols = ['coff.machine', 'optional.subsystem', 'optional.magic']
combined_data = pd.concat([X_train, X_unseen])
combined_data_encoded = pd.get_dummies(combined_data, columns=nominal_cols)

X_train_encoded = combined_data_encoded.iloc[:len(X_train)]
X_unseen_encoded = combined_data_encoded.iloc[len(X_train):]

# Feature scaling and imputation
imputer = IterativeImputer(random_state=42)
scaler = StandardScaler()
selector = VarianceThreshold()

# Apply imputer and scaler to training data
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Feature selection using SelectKBest
select = SelectKBest(f_classif, k=2)  # Adjust 'k' as needed
X_train_selected = select.fit_transform(X_train_scaled, y_train)

# Define the best hyperparameters for your GradientBoostingClassifier
best_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}

# Apply the same transformations to unseen data
X_unseen_imputed = imputer.transform(X_unseen_encoded)
X_unseen_scaled = scaler.transform(X_unseen_imputed)
X_unseen_selected = select.transform(X_unseen_scaled)

# Train the model
model = GradientBoostingClassifier(**best_params, random_state=42)
model.fit(X_train_selected, y_train)

# Predict on unseen data
y_unseen_pred = model.predict(X_unseen_selected)

# Output predictions
for pred in y_unseen_pred:
    print(int(pred))