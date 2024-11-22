# scripts/preprocess.py

import pandas as pd
import numpy as np
import os
import json
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")

def preprocess_data(subset=False, samples_per_class=100):
    """
    Preprocesses the network intrusion detection data.

    Parameters:
    - subset (bool): If True, creates a subset of the data with an even distribution of classes.
    - samples_per_class (int): Number of samples per class in the subset.

    Returns:
    - train_df (pd.DataFrame): Training dataframe.
    - test_df (pd.DataFrame): Testing dataframe.
    - feature_cols (list): List of feature column names.
    - label_dict (dict): Dictionary mapping labels to integers.
    """
    # Define the directory containing the CSV files
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
    
    # Initialize list to collect processed chunks
    chunk_list = []
    
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"Processing file in chunks: {file_path}")
        chunk_size = 100000  # Adjust based on memory
        
        try:
            # Read the CSV file in chunks
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                # Replace infinity values with NaN
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                # Drop rows with NaN values
                chunk.dropna(inplace=True)
                # Append the cleaned chunk to the list
                chunk_list.append(chunk)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue  # Skip to the next file if there's an error
    
    # Concatenate all cleaned chunks into a single DataFrame
    print("Concatenating all cleaned chunks...")
    df_all = pd.concat(chunk_list, axis=0)
    print(f"Combined DataFrame shape: {df_all.shape}")
    del chunk_list  # Free up memory
    
    # Data Cleaning: Drop unnecessary columns
    print("Dropping unnecessary columns...")
    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
    columns_to_drop = [col for col in columns_to_drop if col in df_all.columns]
    df_all.drop(columns=columns_to_drop, inplace=True)
    print(f"Data shape after dropping columns: {df_all.shape}")
    
    # Fix data types
    print("Fixing data types...")
    df_all = fix_data_type(df_all)
    
    # Generate binary labels
    print("Generating binary labels...")
    df_all = generate_binary_label(df_all)
    
    # Reduce memory usage
    print("Reducing memory usage...")
    df_all, _ = reduce_mem_usage(df_all)
    
    # Transform multi-labels
    print("Transforming multi-labels...")
    df_all = transform_multi_label(df_all)
    
    # Balance data
    print("Balancing data...")
    df_all = balance_data(df_all)
    
    # If subset flag is True, create a subset with even class distribution
    if subset:
        print(f"Creating a subset with {samples_per_class} samples per class...")
        df_all = create_subset(df_all, samples_per_class)
        print(f"Subset DataFrame shape: {df_all.shape}")
        print("Subset label distribution:\n", df_all['Label'].value_counts())
    
    # Drop constant and duplicate columns
    print("Dropping constant and duplicate columns...")
    df_all = drop_constant_duplicate_columns(df_all)
    
    # Define feature columns AFTER dropping duplicates
    feature_cols = [col for col in df_all.columns if col not in ['Label', 'Threat']]
    print(f"Feature columns: {feature_cols}")
    
    # Ensure all feature columns are numeric
    print("Converting feature columns to numeric...")
    df_all[feature_cols] = df_all[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # Check for problematic values before scaling
    print("Checking for problematic values before scaling...")
    print("Infinite values count:\n", np.isinf(df_all[feature_cols]).sum())
    print("NaN values count:\n", df_all[feature_cols].isna().sum())
    print("Top 10 largest values per column:\n", df_all[feature_cols].max().sort_values(ascending=False).head(10))
    
    # Replace any remaining infinities with NaN
    df_all[feature_cols] = df_all[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values (either drop or fill)
    print("Handling NaN values...")
    # Option 1: Drop rows with NaN in feature columns
    df_all.dropna(subset=feature_cols, inplace=True)
    # Option 2: Alternatively, fill NaN with column means (uncomment if preferred)
    # df_all[feature_cols] = df_all[feature_cols].fillna(df_all[feature_cols].mean())
    
    # Cap excessively large values at the 99th percentile
    print("Capping excessively large values...")
    cap = df_all[feature_cols].quantile(0.99)
    df_all[feature_cols] = df_all[feature_cols].clip(upper=cap, axis=1)
    
    # Scaling features
    print("Scaling features...")
    scaler = MinMaxScaler()
    df_all[feature_cols] = scaler.fit_transform(df_all[feature_cols])
    
    # Split into train and test sets
    print("Splitting into train and test sets...")
    train_df, test_df = train_test_split(
        df_all, test_size=0.2, random_state=2, shuffle=True, stratify=df_all['Label']
    )
    
    # Save processed data
    print("Saving processed data...")
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_df.csv", index=False)
    test_df.to_csv("data/test_df.csv", index=False)
    
    # Create and handle subset if requested
    if subset:
        print(f"Creating subset files with {samples_per_class} samples per class...")
        train_df_subset = create_subset(train_df, samples_per_class)
        test_df_subset = create_subset(test_df, samples_per_class)
        
        # Save subset files
        train_df_subset.to_csv("data/train_df_subset.csv", index=False)
        test_df_subset.to_csv("data/test_df_subset.csv", index=False)
        print(f"Subset files created - Train shape: {train_df_subset.shape}, Test shape: {test_df_subset.shape}")
    
    # Create and save label dictionary
    print("Creating label dictionary...")
    label_dict = {v: i for i, v in enumerate(df_all['Label'].unique())}
    os.makedirs("data/mappings", exist_ok=True)
    with open("data/mappings/label_dict.json", "w") as outfile:
        json.dump(label_dict, outfile)
    
    print("Data preprocessing completed successfully.")
    
    return train_df, test_df, feature_cols, label_dict

def fix_data_type(df):
    print("Fixing data types...")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    print("Data types fixed.")
    return df

def generate_binary_label(df):
    print("Generating binary labels...")
    df['Threat'] = df['Label'].apply(lambda x: "Benign" if x == 'Benign' else "Malicious")
    print("Threat categories:", df['Threat'].unique())
    print("Threat distribution:\n", df['Threat'].value_counts())
    return df

def reduce_mem_usage(df):
    print("Reducing memory usage...")
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after: {end_mem:.2f} MB")
    print(f"Reduced by {(100 * (start_mem - end_mem) / start_mem):.1f}%")
    
    return df, None

def transform_multi_label(df):
    print("Transforming multi-labels...")
    mapping= {
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
        'Label': 'Benign'
    }
    df['Label'] = df['Label'].map(mapping)
    print("Transformed labels:")
    print(df['Label'].value_counts())
    return df

def balance_data(df):
    print("Balancing data...")
    X = df.drop(["Label"], axis=1)
    y = df["Label"]
    
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X, y)
    
    df_balanced = pd.concat([X_balanced, y_balanced], axis=1)
    print(f"Balanced data shape: {df_balanced.shape}")
    print("Balanced label distribution:\n", df_balanced['Label'].value_counts())
    
    return df_balanced

def create_subset(df, samples_per_class):
    """
    Creates a subset of the dataframe with an even distribution of classes.

    Parameters:
    - df (pd.DataFrame): The balanced dataframe.
    - samples_per_class (int): Number of samples per class.

    Returns:
    - subset_df (pd.DataFrame): Subset dataframe.
    """
    classes = df['Label'].unique()
    subset_list = []
    
    for cls in classes:
        cls_df = df[df['Label'] == cls]
        if len(cls_df) >= samples_per_class:
            sampled_df = cls_df.sample(n=samples_per_class, random_state=42)
        else:
            sampled_df = cls_df  # If not enough samples, take all
            print(f"Warning: Not enough samples for class '{cls}'. Expected {samples_per_class}, got {len(cls_df)}.")
        subset_list.append(sampled_df)
    
    subset_df = pd.concat(subset_list, axis=0).reset_index(drop=True)
    return subset_df

def drop_constant_duplicate_columns(df):
    print("Dropping constant columns...")
    # Drop constant columns
    variances = df.var(numeric_only=True)
    constant_columns = variances[variances == 0].index
    if len(constant_columns) > 0:
        df.drop(columns=constant_columns, inplace=True)
        print(f"Dropped constant columns: {list(constant_columns)}")
    else:
        print("No constant columns found.")
    
    print("Dropping duplicate columns...")
    # Drop duplicate columns
    duplicates = set()
    for i in range(len(df.columns)):
        col1 = df.columns[i]
        for j in range(i+1, len(df.columns)):
            col2 = df.columns[j]
            if df[col1].equals(df[col2]):
                duplicates.add(col2)
    if len(duplicates) > 0:
        df.drop(columns=duplicates, inplace=True)
        print(f"Dropped duplicate columns: {list(duplicates)}")
    else:
        print("No duplicate columns found.")
    
    print(f"Data shape after dropping columns: {df.shape}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess network intrusion detection data.")
    parser.add_argument('--subset', action='store_true', help='Create a subset of the data with even class distribution.')
    parser.add_argument('--samples_per_class', type=int, default=100, help='Number of samples per class in the subset.')
    
    args = parser.parse_args()
    
    preprocess_data(subset=args.subset, samples_per_class=args.samples_per_class)