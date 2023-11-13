import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold

# Load the dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

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
#select = SelectKBest(score_func=f_classif, k=2)  # Adjust 'k' as needed
select = SelectKBest(score_func=f_classif, k=2)  # Adjust 'k' as needed
X_train_selected = select.fit_transform(X_train, y_train)
X_val_selected = select.transform(X_val)

ensemble_model = VotingClassifier(estimators=[('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier()), ('knn', KNeighborsClassifier())], voting='soft')

class_weights = {0: 2, 1: 1}
# Initialize a list of models
models = [
    RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=42),  # Default parameters for Decision Tree
    SVC(probability=True),  # Probability for soft voting in the ensemble
    MLPClassifier(max_iter=500, early_stopping=True, random_state=42)
]

# Train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train and evaluate models
for model in models:
    # Train the model
    model.fit(X_train_selected, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val_selected)
    
    # Evaluation
    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_val, y_val_pred))

# Cross-validation for the best model
# Example: GradientBoostingClassifier
best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='f1_macro')
print(f"Cross-validated F1 scores: {scores}")
