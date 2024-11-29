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
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
import warnings
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_and_clean_data(data_dir, csv_files):
    """
    - data_dir (str): Directory containing the CSV files.
    - csv_files (list): List of CSV file names to load.

    Returns:
    - df_all (pd.DataFrame): Concatenated and cleaned DataFrame.
    """
    chunk_list = []
    for file in tqdm(csv_files, desc="Loading files"):
        file_path = os.path.join(data_dir, file)
        print(f"Loading file: {file_path}")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"Original shape: {df.shape}")

            # Drop unnecessary columns if they exist, but keep Timestamp
            ##columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            #if columns_to_drop:
            #    df.drop(columns=columns_to_drop, inplace=True)
            #    print(f"Dropped columns: {columns_to_drop}")
            

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

            # Drop any remaining non-numeric columns except 'Label', 'Threat', and 'Timestamp'
            non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
            columns_to_exclude = ['Label', 'Threat', 'Timestamp']  # Added Timestamp to excluded columns
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

    # Remove duplicate columns (excluding 'Label' and 'Threat')
    duplicates = set()
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if cols[j] in duplicates or cols[j] in ['Label', 'Threat']:
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

    Parameters:
    - df (pd.DataFrame): DataFrame to process.
    - threshold (float): Correlation threshold above which to remove features.

    Returns:
    - df (pd.DataFrame): DataFrame with highly correlated features removed.
    """
    print("\nCalculating Pearson correlation matrix...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Use tqdm for correlation calculation
    corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)
    for col1 in tqdm(numeric_cols, desc="Computing correlations"):
        for col2 in numeric_cols:
            if col2 >= col1:  # Only compute upper triangle
                corr = df[col1].corr(df[col2])
                corr_matrix.loc[col1, col2] = corr
                corr_matrix.loc[col2, col1] = corr
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"Dropped highly correlated columns (>{threshold} correlation): {to_drop}")
    else:
        print("No highly correlated columns to drop.")

    print(f"DataFrame shape after removing highly correlated features: {df.shape}")
    return df

def split_data(df, label_col='Label', threat_col='Threat', test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets with stratification.

    Parameters:
    - df (pd.DataFrame): DataFrame to split.
    - label_col (str): Name of the label column.
    - threat_col (str): Name of the binary threat column.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed.

    Returns:
    - train_df (pd.DataFrame): Training set.
    - test_df (pd.DataFrame): Testing set.
    """
    print("\nSplitting data into train and test sets...")
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True, stratify=df[label_col]
    )
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    print("\nTraining set distribution:")
    print(train_df[label_col].value_counts())
    print("\nTesting set distribution:")
    print(test_df[label_col].value_counts())
    return train_df, test_df

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
    print("\nScaling training set features...")
    for col in tqdm(feature_cols, desc="Scaling features"):
        train_df[col] = scaler.fit_transform(train_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])
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

def save_processed_data(train_df, test_df, train_subset_df, feature_cols, selected_features, label_dict, class_weights, output_dir='processed_data'):
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

def create_hierarchical_datasets(train_df_balanced, test_df, feature_cols, output_dir='processed_data', subset_size=100):
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
    test_binary = test_df[['Threat'] + feature_cols]
    train_binary.to_csv(os.path.join(binary_output_dir, 'train_binary.csv'), index=False)
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
    test_multi = test_malicious[['Label'] + feature_cols]
    train_multi.to_csv(os.path.join(multi_class_output_dir, 'train_multi_class.csv'), index=False)
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

def sort_by_timestamp(df):
    """Sort dataframe by timestamp to maintain temporal order."""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.sort_values('Timestamp').reset_index(drop=True)

def create_time_features(df):
    """Create additional time-based features."""
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['time_of_day'] = df['hour'] * 60 + df['minute']
    return df

def create_sequences(df, sequence_length=10, stride=1):
    """Create sequences of data for temporal analysis."""
    print("\nCreating sequences...")
    total_sequences = (len(df) - sequence_length + 1) // stride
    sequences = []
    labels = []
    timestamps = []
    
    # Use tqdm for sequence creation
    with tqdm(total=total_sequences, desc="Creating sequences") as pbar:
        for i in range(0, len(df) - sequence_length + 1, stride):
            sequence = df.iloc[i:i + sequence_length]
            feature_sequence = sequence.drop(['Label', 'Threat'], axis=1).values
            timestamp = sequence.iloc[-1]['Timestamp']
            label = sequence.iloc[-1]['Label']
            
            sequences.append(feature_sequence)
            labels.append(label)
            timestamps.append(timestamp)
            pbar.update(1)
    
    print("\nSequence creation completed.")
    return np.array(sequences), np.array(labels), np.array(timestamps)

def track_feature_changes(original_columns, final_columns):
    """
    Track which features were kept and dropped during preprocessing.
    
    Parameters:
    - original_columns (list): Original feature columns
    - final_columns (list): Final feature columns after preprocessing
    
    Returns:
    - dict: Dictionary containing kept and dropped features
    """
    # Remove non-feature columns from comparison
    non_feature_cols = ['Label', 'Threat', 'Timestamp', 'hour', 'minute', 'day_of_week', 'time_of_day']
    original_features = [col for col in original_columns if col not in non_feature_cols]
    final_features = [col for col in final_columns if col not in non_feature_cols]
    
    # Find kept and dropped features
    kept_features = set(final_features)
    dropped_features = set(original_features) - kept_features
    
    return {
        'kept': sorted(list(kept_features)),
        'dropped': sorted(list(dropped_features))
    }

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
    feature_importance_threshold = 0.01

    # Load and clean data
    df_all = load_and_clean_data(data_dir, csv_files)

    # Store original columns for comparison AFTER loading data
    original_columns = df_all.columns.tolist()

    # Sort by timestamp (critical for temporal order)
    df_all = sort_by_timestamp(df_all)
    
    # Create time features
    df_all = create_time_features(df_all)
    
    # Transform labels
    df_all = transform_labels(df_all)

    # Remove constant and duplicate columns
    df_all = remove_constant_and_duplicate_columns(df_all)

    # Remove highly correlated features
    df_all = remove_highly_correlated_features(df_all, threshold=0.90)

    # Create sequences before train/test split to maintain temporal continuity
    sequence_length = 10
    stride = 5
    X_sequences, y_sequences, timestamps = create_sequences(df_all, sequence_length, stride)
    
    # Split sequences into train/test maintaining temporal order
    split_idx = int(len(X_sequences) * 0.8)
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
    train_timestamps, test_timestamps = timestamps[:split_idx], timestamps[split_idx:]
    
    # Convert sequences back to DataFrames with timestamps
    train_df = pd.DataFrame({
        'Label': y_train,
        'Timestamp': train_timestamps
    })
    
    test_df = pd.DataFrame({
        'Label': y_test,
        'Timestamp': test_timestamps
    })
    
    # Add sequence features to DataFrames
    for i in range(X_train.shape[2]):
        train_df[f'feature_{i}'] = X_train[:, -1, i]
        test_df[f'feature_{i}'] = X_test[:, -1, i]
    
    # Feature columns
    feature_cols = [col for col in train_df.columns if col not in [label_col, threat_col, 'Timestamp']]

    # Scale features
    train_df, test_df, scaler = scale_features(train_df, test_df, feature_cols)

    # Create hierarchical datasets
    create_hierarchical_datasets(
        train_df_balanced=train_df,  # Note: Using unbalanced data
        test_df=test_df,
        feature_cols=feature_cols,
        output_dir='processed_data'
    )

    # Track feature changes
    final_columns = train_df.columns.tolist()
    feature_changes = track_feature_changes(original_columns, final_columns)
    
    # Print feature change summary
    print("\nFeature Processing Summary:")
    print(f"\nTotal original features: {len(original_columns)}")
    print(f"Total final features: {len(final_columns)}")
    print(f"\nKept {len(feature_changes['kept'])} features:")
    for feature in feature_changes['kept']:
        print(f"  - {feature}")
    print(f"\nDropped {len(feature_changes['dropped'])} features:")
    for feature in feature_changes['dropped']:
        print(f"  - {feature}")

    print("\nPreprocessing completed successfully.")

if __name__ == "__main__":
    main()

