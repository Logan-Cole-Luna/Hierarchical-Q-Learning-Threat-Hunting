"""
Data preprocessing pipeline for network intrusion detection.

This script handles the complete preprocessing pipeline including:
1. Data loading and cleaning
2. Feature engineering and selection
3. Label transformation
4. Train/test splitting
5. Data balancing
6. Feature scaling
7. Creation of hierarchical datasets

Key Features:
- Handles both binary and multi-class classification
- Removes constant and highly correlated features
- Supports multiple balancing methods (SMOTE, undersampling)
- Creates stratified train/test splits
- Generates subset data for quick testing
- Saves preprocessed data and metadata

Functions:
    Multiple preprocessing and utility functions documented within the script.
"""

import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
import warnings

warnings.filterwarnings("ignore")

def load_and_clean_data(data_dir, csv_files):
    """
    Loads multiple CSV files, cleans them by handling infinities and NaNs,
    drops unnecessary columns, removes duplicated header rows, and fixes data types.

    Parameters:
    - data_dir (str): Directory containing the CSV files.
    - csv_files (list): List of CSV file names to load.

    Returns:
    - df_all (pd.DataFrame): Concatenated and cleaned DataFrame.
    """
    chunk_list = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading file: {file_path}")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"Original shape: {df.shape}")

            # Drop unnecessary columns if they exist
            columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
            columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                print(f"Dropped columns: {columns_to_drop}")

            # Replace infinity values with NaN and drop rows with NaNs
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            initial_shape = df.shape
            df.dropna(inplace=True)
            print(f"Dropped rows with NaNs: {initial_shape[0] - df.shape[0]}")

            # Remove duplicated header rows if any
            if 'Label' in df.columns:
                duplicated_headers = df[df['Label'] == 'Label']
                if not duplicated_headers.empty:
                    df = df[df['Label'] != 'Label']
                    print(f"Removed {duplicated_headers.shape[0]} duplicated header rows.")
                else:
                    print("No duplicated header rows found.")
            else:
                print("'Label' column not found. Skipping duplicated header removal.")

            # Fix data types
            df = fix_data_type(df)

            # Drop any remaining non-numeric columns except 'Label' and 'Threat'
            non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
            columns_to_exclude = ['Label', 'Threat']
            non_numeric_cols = [col for col in non_numeric_cols if col not in columns_to_exclude]
            if non_numeric_cols:
                df.drop(columns=non_numeric_cols, inplace=True)
                print(f"Dropped non-numeric columns: {non_numeric_cols}")
            else:
                print("No non-numeric columns to drop.")

            chunk_list.append(df)
            print(f"Processed shape: {df.shape}\n")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not chunk_list:
        raise ValueError("No data loaded. Please check the CSV files and paths.")

    # Concatenate all cleaned chunks
    print("Concatenating all cleaned data...")
    df_all = pd.concat(chunk_list, axis=0).reset_index(drop=True)
    print(f"Combined DataFrame shape: {df_all.shape}")
    
    # Remove duplicate rows
    print("Removing duplicate rows...")
    initial_shape = df_all.shape
    df_all.drop_duplicates(inplace=True)
    duplicates_removed = initial_shape[0] - df_all.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")
    
    del chunk_list  # Free up memory
    return df_all

def fix_data_type(df):
    """
    Converts columns to appropriate data types to optimize memory usage.

    Parameters:
    - df (pd.DataFrame): DataFrame with raw data.

    Returns:
    - df (pd.DataFrame): DataFrame with optimized data types.
    """
    # Define the desired data types for each column
    dtype_mapping = {
        'Dst Port': 'int32',
        'Protocol': 'int8',
        'Flow Duration': 'int32',
        'Tot Fwd Pkts': 'int32',
        'Tot Bwd Pkts': 'int32',
        'TotLen Fwd Pkts': 'int32',
        'TotLen Bwd Pkts': 'int32',
        'Fwd Pkt Len Max': 'int16',
        'Fwd Pkt Len Min': 'int16',
        'Fwd Pkt Len Mean': 'float32',
        'Fwd Pkt Len Std': 'float32',
        'Bwd Pkt Len Max': 'int16',
        'Bwd Pkt Len Min': 'int16',
        'Bwd Pkt Len Mean': 'float32',
        'Bwd Pkt Len Std': 'float32',
        'Flow Byts/s': 'float32',
        'Flow Pkts/s': 'float32',
        'Flow IAT Mean': 'float32',
        'Flow IAT Std': 'float32',
        'Flow IAT Max': 'int32',
        'Flow IAT Min': 'int32',
        'Fwd IAT Tot': 'int32',
        'Fwd IAT Mean': 'float32',
        'Fwd IAT Std': 'float32',
        'Fwd IAT Max': 'int32',
        'Fwd IAT Min': 'int32',
        'Bwd IAT Tot': 'int32',
        'Bwd IAT Mean': 'float32',
        'Bwd IAT Std': 'float32',
        'Bwd IAT Max': 'int32',
        'Bwd IAT Min': 'int32',
        'Fwd PSH Flags': 'int8',
        'Bwd PSH Flags': 'int8',
        'Fwd URG Flags': 'int8',
        'Bwd URG Flags': 'int8',
        'Fwd Header Len': 'int16',
        'Bwd Header Len': 'int16',
        'Fwd Pkts/s': 'float32',
        'Bwd Pkts/s': 'float32',
        'Pkt Len Min': 'int16',
        'Pkt Len Max': 'int16',
        'Pkt Len Mean': 'float32',
        'Pkt Len Std': 'float32',
        'Pkt Len Var': 'float32',
        'FIN Flag Cnt': 'int8',
        'SYN Flag Cnt': 'int8',
        'RST Flag Cnt': 'int8',
        'PSH Flag Cnt': 'int8',
        'ACK Flag Cnt': 'int8',
        'URG Flag Cnt': 'int8',
        'CWE Flag Count': 'int8',
        'ECE Flag Cnt': 'int8',
        'Down/Up Ratio': 'float32',  # Changed to float since ratio can be non-integer
        'Pkt Size Avg': 'float32',
        'Fwd Seg Size Avg': 'float32',
        'Bwd Seg Size Avg': 'float32',
        'Fwd Byts/b Avg': 'float32',
        'Fwd Pkts/b Avg': 'float32',
        'Fwd Blk Rate Avg': 'float32',
        'Bwd Byts/b Avg': 'float32',
        'Bwd Pkts/b Avg': 'float32',
        'Bwd Blk Rate Avg': 'float32',
        'Subflow Fwd Pkts': 'int16',
        'Subflow Fwd Byts': 'int32',
        'Subflow Bwd Pkts': 'int16',
        'Subflow Bwd Byts': 'int32',
        'Init Fwd Win Byts': 'int32',
        'Init Bwd Win Byts': 'int32',
        'Fwd Act Data Pkts': 'int16',
        'Fwd Seg Size Min': 'int16',
        'Active Mean': 'float32',
        'Active Std': 'float32',
        'Active Max': 'int32',
        'Active Min': 'int32',
        'Idle Mean': 'float32',
        'Idle Std': 'float32',
        'Idle Max': 'int32',
        'Idle Min': 'int32',
    }

    for col, dtype in dtype_mapping.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except ValueError as e:
                print(f"Error converting column {col} to {dtype}: {e}")
                # Optionally, drop or impute problematic columns
                df.drop(columns=col, inplace=True)
                print(f"Dropped column {col} due to conversion error.")

    return df

def transform_labels(df):
    """
    Transforms multi-labels into broader categories and generates a binary 'Threat' label.

    Parameters:
    - df (pd.DataFrame): DataFrame with raw labels.

    Returns:
    - df (pd.DataFrame): DataFrame with transformed labels.
    """
    mapping = {
        'SSH-Bruteforce': 'Brute-force',
        'FTP-BruteForce': 'Brute-force',
        'Brute Force -XSS': 'Web attack',
        'Brute Force -Web': 'Web attack',
        'SQL Injection': 'Web attack',
        'DoS attacks-Hulk': 'DoS attack',
        'DoS attacks-SlowHTTPTest': 'DoS attack',
        'DoS attacks-Slowloris': 'DoS attack',
        'DoS attacks-GoldenEye': 'DoS attack',
        'DDOS attack-HOIC': 'DDoS attack',
        'DDOS attack-LOIC-UDP': 'DDoS attack',
        'DDoS attacks-LOIC-HTTP': 'DDoS attack',
        'Bot': 'Botnet',
        'Infilteration': 'Infilteration',
        'Benign': 'Benign',
        'Label': 'Benign',  # Assuming 'Label' can sometimes be 'Benign'
    }

    df['Label'] = df['Label'].map(mapping)

    # Generate binary 'Threat' label
    df['Threat'] = df['Label'].apply(lambda x: 'Benign' if x == 'Benign' else 'Malicious')

    print("\nTransformed Labels:")
    print(df['Label'].value_counts())
    print("\nThreat Distribution:")
    print(df['Threat'].value_counts())

    return df

def remove_constant_and_duplicate_columns(df):
    """
    Removes constant and duplicate columns from the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to process.

    Returns:
    - df (pd.DataFrame): DataFrame with constant and duplicate columns removed.
    """
    # Remove constant columns (excluding 'Label' and 'Threat')
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1 and col not in ['Label', 'Threat']]
    if constant_columns:
        df.drop(columns=constant_columns, inplace=True)
        print(f"Dropped constant columns: {constant_columns}")
    else:
        print("No constant columns to drop.")

    # Exclude 'Pkt Len Min' from being removed as duplicate
    duplicates = set()
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if cols[j] in duplicates or cols[j] in ['Label', 'Threat', 'Pkt Len Min']:
                continue
            if df[cols[i]].equals(df[cols[j]]):
                duplicates.add(cols[j])

    if duplicates:
        df.drop(columns=duplicates, inplace=True)
        print(f"Dropped duplicate columns: {duplicates}")
    else:
        print("No duplicate columns to drop.")

    print(f"DataFrame shape after removing constant and duplicate columns: {df.shape}")
    return df

def remove_highly_correlated_features(df, threshold=0.90):
    """
    Removes highly correlated features based on Pearson correlation.
    """
    print("\nCalculating Pearson correlation matrix...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Exclude 'Pkt Len Min' from being considered for dropping
    to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column != 'Pkt Len Min']

    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"Dropped highly correlated columns (>{threshold} correlation): {to_drop}")
    else:
        print("No highly correlated columns to drop.")

    print(f"DataFrame shape after removing highly correlated features: {df.shape}")
    return df

def split_data(df, label_col='Label', threat_col='Threat', test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits the DataFrame into training, validation, and testing sets with stratification.
    """
    print("\nSplitting data into train and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True, stratify=df[label_col]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    print("\nTraining set distribution:")
    print(train_df[label_col].value_counts())
    print("\nTesting set distribution:")
    print(test_df[label_col].value_counts())
    
    # Check for duplicates between train and test sets
    train_hashes = train_df.apply(lambda row: hash(tuple(row)), axis=1)
    test_hashes = set(test_df.apply(lambda row: hash(tuple(row)), axis=1))
    overlap = train_hashes.isin(test_hashes).sum()
    print(f"\nNumber of overlapping records between train and test sets: {overlap}")

    # Split training set into training and validation sets
    print("\nSplitting training data into training and validation sets...")
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state, shuffle=True, stratify=train_df[label_col]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Updated Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print("\nUpdated Training set distribution:")
    print(train_df[label_col].value_counts())
    print("\nValidation set distribution:")
    print(val_df[label_col].value_counts())
    
    return train_df, val_df, test_df

def balance_training_set(train_df, label_col='Label', method='smote', random_state=42, samples_per_class=None):
    """
    Balances the training set using the specified method.

    Parameters:
    - train_df (pd.DataFrame): Training set.
    - label_col (str): Name of the label column.
    - method (str): Balancing method ('smote' or 'undersample').
    - random_state (int): Random seed.
    - samples_per_class (int or None): If using undersampling with a fixed number of samples per class.

    Returns:
    - train_df_balanced (pd.DataFrame): Balanced training set.
    """
    # Keep track of the Threat column
    X = train_df.drop([label_col], axis=1)  # Keep 'Threat' column in X
    y = train_df[label_col]

    print("\nOriginal training set class distribution:")
    print(y.value_counts())

    if method == 'smote':
        print("\nApplying SMOTE for oversampling minority classes...")
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == 'undersample':
        print("\nApplying Random UnderSampling to balance classes...")
        rus = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = rus.fit_resample(X, y)
    elif method == 'none':
        print("\nNo balancing applied to the training set.")
        train_df_balanced = train_df.copy()
        print(f"Training set shape remains: {train_df_balanced.shape}")
        return train_df_balanced
    else:
        raise ValueError("Unsupported balancing method. Choose 'smote', 'undersample', or 'none'.")

    # Reconstruct DataFrame with both Label and Threat
    train_df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    train_df_balanced[label_col] = y_balanced
    
    print(f"\nBalanced training set shape: {train_df_balanced.shape}")
    print(f"Balanced training set class distribution:\n{train_df_balanced[label_col].value_counts()}")

    return train_df_balanced

def scale_features(train_df, test_df, feature_cols):
    """
    Scales features using MinMaxScaler fitted on the training set.

    Parameters:
    - train_df (pd.DataFrame): Training set.
    - test_df (pd.DataFrame): Testing set.
    - feature_cols (list): List of feature column names.

    Returns:
    - train_df (pd.DataFrame): Scaled training set.
    - test_df (pd.DataFrame): Scaled testing set.
    - scaler (MinMaxScaler): Fitted scaler object.
    """
    print("\nScaling features with MinMaxScaler...")
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print("Feature scaling completed.")
    return train_df, test_df, scaler

def compute_class_weights(train_df, label_col='Label'):
    """
    Computes class weights based on the training set for the specified label column.
    """
    print(f"\nComputing class weights for '{label_col}'...")
    classes = np.unique(train_df[label_col])
    y = train_df[label_col].values
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {label: weight for label, weight in zip(classes, weights)}
    print(f"Class Weights ({label_col}): {class_weights}")
    return class_weights

def encode_labels(train_df, test_df, label_col='Label'):
    """
    Encodes labels into integers and creates a label dictionary for the specified label column.
    """
    print(f"\nEncoding labels for '{label_col}'...")
    classes = np.unique(train_df[label_col])
    label_dict = {label: idx for idx, label in enumerate(classes)}
    print(f"Label Dictionary ({label_col}): {label_dict}")
    train_df[label_col] = train_df[label_col].map(label_dict)
    test_df[label_col] = test_df[label_col].map(label_dict)
    return label_dict

def save_label_dict_and_class_weights(label_dict, class_weights, output_dir):
    """
    Saves the label dictionary and class weights to the specified output directory.
    """
    # Save label dictionary
    with open(os.path.join(output_dir, 'label_dict.json'), 'w') as f:
        json.dump(label_dict, f)
    print(f"Saved label dictionary to '{output_dir}/label_dict.json'.")

    # Save class weights
    with open(os.path.join(output_dir, 'class_weights.json'), 'w') as f:
        json.dump(class_weights, f)
    print(f"Saved class weights to '{output_dir}/class_weights.json'.")

def train_xgb_and_select_features(train_df, test_df, feature_cols, y_train, y_test, label_dict, output_dir='processed_data', importance_threshold=0.01):
    """
    Trains an XGBoost classifier, evaluates it, and selects important features based on feature importances.

    Parameters:
    - train_df (pd.DataFrame): Balanced training set.
    - test_df (pd.DataFrame): Testing set.
    - feature_cols (list): List of feature column names.
    - y_train (list): Encoded training labels.
    - y_test (list): Encoded testing labels.
    - label_dict (dict): Dictionary mapping labels to indices.
    - output_dir (str): Directory to save feature-selected data.
    - importance_threshold (float): Threshold for feature importances to select features.

    Returns:
    - selected_features (list): List of selected feature names.
    """
    print("\nTraining XGBoost classifier for feature selection...")
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(train_df[feature_cols].values, y_train)

    y_pred = model.predict(test_df[feature_cols].values)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df)

    # Select features based on importance threshold
    selected_features = importance_df[importance_df['Importance'] >= importance_threshold]['Feature'].tolist()
    print(f"\nSelected Features (Importance >= {importance_threshold}): {selected_features}")

    # Save feature importances
    os.makedirs(output_dir, exist_ok=True)
    importance_df.to_csv(os.path.join(output_dir, 'xgb_feature_importances.csv'), index=False)
    print(f"Saved feature importances to '{output_dir}/xgb_feature_importances.csv'.")

    return selected_features

def create_train_subset(train_df_balanced, label_col='Label', samples_per_class=10, random_state=42):
    """
    Creates a subset of the training data with a fixed number of samples per class.

    Parameters:
    - train_df_balanced (pd.DataFrame): Balanced training set.
    - label_col (str): Name of the label column.
    - samples_per_class (int): Number of samples per class in the subset.
    - random_state (int): Random seed.

    Returns:
    - train_subset_df (pd.DataFrame): Subset of the training set.
    """
    print(f"\nCreating a training subset with {samples_per_class} samples per class for overfitting test...")
    train_subset_df = train_df_balanced.groupby(label_col).apply(
        lambda x: x.sample(n=samples_per_class, random_state=random_state) if len(x) >= samples_per_class else x
    ).reset_index(drop=True)
    print(f"Training subset shape: {train_subset_df.shape}")
    print(f"Training subset distribution:\n{train_subset_df[label_col].value_counts()}")
    return train_subset_df

def save_processed_data(train_df, val_df, test_df, train_subset_df, feature_cols, selected_features, label_dict, class_weights, output_dir='processed_data'):
    """
    Saves the processed training, testing, and subset datasets, along with label dictionary and class weights.

    Parameters:
    - train_df (pd.DataFrame): Balanced training set.
    - test_df (pd.DataFrame): Testing set.
    - train_subset_df (pd.DataFrame): Subset of the training set for overfitting test.
    - feature_cols (list): List of feature column names.
    - selected_features (list): List of selected feature names after feature selection.
    - label_dict (dict): Dictionary mapping labels to indices.
    - class_weights (dict): Dictionary mapping class indices to weights.
    - output_dir (str): Directory to save the processed data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add 'Threat' column to selected features if it's not already included
    save_columns = selected_features + ['Label']
    if 'Threat' in train_df.columns:
        save_columns.append('Threat')

    # Save training and testing sets with selected features
    train_df[save_columns].to_csv(os.path.join(output_dir, 'train_df.csv'), index=False)
    test_df[save_columns].to_csv(os.path.join(output_dir, 'test_df.csv'), index=False)
    print(f"\nSaved processed training and testing sets to '{output_dir}' directory.")

    # Save training subset
    train_subset_df[save_columns].to_csv(os.path.join(output_dir, 'train_subset_df.csv'), index=False)
    print(f"Saved training subset to '{output_dir}/train_subset_df.csv'.")

    # Save label dictionary
    with open(os.path.join(output_dir, 'label_dict.json'), 'w') as f:
        json.dump(label_dict, f)
    print(f"Saved label dictionary to '{output_dir}/label_dict.json'.")

    # Save class weights
    with open(os.path.join(output_dir, 'class_weights.json'), 'w') as f:
        json.dump(class_weights, f)
    print(f"Saved class weights to '{output_dir}/class_weights.json'.")

def create_hierarchical_datasets(train_df_balanced, val_df, test_df, feature_cols, output_dir='processed_data', subset_size=100):
    """
    Creates hierarchical datasets for binary and multi-class classification,
    including subsets for overfitting tests, and saves separate label dictionaries and class weights.
    """
    print("\nCreating hierarchical datasets for binary and multi-class classification...")
    
    # 1. Binary Classification Dataset
    binary_output_dir = os.path.join(output_dir, 'binary_classification')
    os.makedirs(binary_output_dir, exist_ok=True)

    # Save binary classification datasets
    train_binary = train_df_balanced[['Threat'] + feature_cols]
    val_binary = val_df[['Threat'] + feature_cols]  # Added validation set
    test_binary = test_df[['Threat'] + feature_cols]
    train_binary.to_csv(os.path.join(binary_output_dir, 'train_binary.csv'), index=False)
    val_binary.to_csv(os.path.join(binary_output_dir, 'val_binary.csv'), index=False)  # Save validation set
    test_binary.to_csv(os.path.join(binary_output_dir, 'test_binary.csv'), index=False)
    print(f"Saved binary classification datasets to '{binary_output_dir}'.")

    # Create binary subset for overfitting tests
    train_binary_subset = train_binary.groupby('Threat', group_keys=False).apply(lambda x: x.sample(n=min(len(x), subset_size)))
    train_binary_subset.to_csv(os.path.join(binary_output_dir, 'train_binary_subset.csv'), index=False)
    print(f"Saved binary classification subset to '{binary_output_dir}/train_binary_subset.csv'.")

    # Compute and save label dictionary and class weights for binary classification
    binary_class_weights = compute_class_weights(train_binary, label_col='Threat')
    binary_label_dict = encode_labels(train_binary, test_binary, label_col='Threat')
    save_label_dict_and_class_weights(binary_label_dict, binary_class_weights, binary_output_dir)

    # 2. Multi-Class Classification Dataset
    multi_class_output_dir = os.path.join(output_dir, 'multi_class_classification')
    os.makedirs(multi_class_output_dir, exist_ok=True)

    # Map 'Malicious' to its encoded value
    malicious_label = next(key for key, value in binary_label_dict.items() if value == binary_label_dict['Malicious'])

    # Filter malicious samples for multi-class classification
    train_malicious = train_df_balanced[train_df_balanced['Threat'] == malicious_label]
    test_malicious = test_df[test_df['Threat'] == malicious_label]

    # Save multi-class classification datasets
    train_multi = train_malicious[['Label'] + feature_cols]
    val_multi = val_df[val_df['Threat'] == 'Malicious'][['Label'] + feature_cols]  # Added validation set
    test_multi = test_malicious[['Label'] + feature_cols]
    train_multi.to_csv(os.path.join(multi_class_output_dir, 'train_multi_class.csv'), index=False)
    val_multi.to_csv(os.path.join(multi_class_output_dir, 'val_multi_class.csv'), index=False)  # Save validation set
    test_multi.to_csv(os.path.join(multi_class_output_dir, 'test_multi_class.csv'), index=False)
    print(f"Saved multi-class classification datasets to '{multi_class_output_dir}'.")

    # Create multi-class subset for overfitting tests
    train_multi_subset = train_multi.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), subset_size)))
    train_multi_subset.to_csv(os.path.join(multi_class_output_dir, 'train_multi_class_subset.csv'), index=False)
    print(f"Saved multi-class classification subset to '{multi_class_output_dir}/train_multi_class_subset.csv'.")

    # Compute and save label dictionary and class weights for multi-class classification
    multi_class_weights = compute_class_weights(train_multi, label_col='Label')
    multi_label_dict = encode_labels(train_multi, test_multi, label_col='Label')
    save_label_dict_and_class_weights(multi_label_dict, multi_class_weights, multi_class_output_dir)

def main():
    # Define parameters
    data_dir = "data/"
    csv_files = [
        "02-14-2018.csv",
        "02-15-2018.csv",
        "02-16-2018.csv",
        "02-20-2018.csv",
        "02-21-2018.csv",
        "02-22-2018.csv",
        "02-23-2018.csv",
        "02-28-2018.csv",
        "03-01-2018.csv",
        "03-02-2018.csv"
    ]
    label_col = "Label"
    threat_col = "Threat"
    test_size = 0.2

    
    random_state = 42
    balancing_method = 'none'  # Options: 'smote', 'undersample', 'none'
    samples_per_class_subset = 10  # Number of samples per class in train_subset_df
    feature_importance_threshold = 0.01  # Threshold for feature selection

    # Load and clean data
    df_all = load_and_clean_data(data_dir, csv_files)

    # Transform labels
    df_all = transform_labels(df_all)

    # Remove constant and duplicate columns
    df_all = remove_constant_and_duplicate_columns(df_all)

    # Remove highly correlated features
    df_all = remove_highly_correlated_features(df_all, threshold=0.90)

    # Split into train, validation, and test sets
    train_df, val_df, test_df = split_data(df_all, label_col=label_col, threat_col=threat_col, test_size=test_size, random_state=random_state)
    del df_all  # Free up memory

    # Feature columns
    feature_cols = [col for col in train_df.columns if col not in [label_col, threat_col]]

    # Balance the training set
    train_df_balanced = balance_training_set(
        train_df,
        label_col=label_col,
        method=balancing_method,
        random_state=random_state
    )
    del train_df  # Free up memory

    # Scale features
    train_df_balanced, test_df, scaler = scale_features(train_df_balanced, test_df, feature_cols)

    # Create hierarchical datasets with subsets
    create_hierarchical_datasets(
        train_df_balanced=train_df_balanced,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        output_dir='processed_data',
        subset_size=100  # Adjust the subset size as needed
    )
    
    print("\nPreprocessing completed successfully.")

if __name__ == "__main__":
    main()
