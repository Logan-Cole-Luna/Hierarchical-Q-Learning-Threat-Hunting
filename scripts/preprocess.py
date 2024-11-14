# scripts/preprocess.py

"""
Data Preprocessing Script for Hierarchical Q-Learning Threat Hunting

This script preprocesses network intrusion detection datasets for hierarchical Q-Learning models.
It performs data loading, cleaning, feature selection, label mapping, normalization, and subset creation.
Both High-Level and Low-Level labels are separated and saved for subsequent training and evaluation.

Usage:
    python preprocess.py

Dependencies:
    - pandas
    - numpy
    - scikit-learn
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class DataPreprocessor:
    """
    Preprocesses network intrusion detection data for hierarchical Q-Learning models.

    Attributes:
        data_dir (str): Directory containing raw CSV data files.
        high_level_features (list): Feature names for the High-Level Q-Network.
        low_level_features (list): Feature names for the Low-Level Q-Network.
        label_column (str): Column name containing labels.
        scaler_high (MinMaxScaler): Scaler for High-Level features.
        scaler_low (MinMaxScaler): Scaler for Low-Level features.
        data (pd.DataFrame): Combined and cleaned dataset.
        mappings (dict): Mappings for categories and anomalies.
    """

    def __init__(self, data_dir, high_level_features, low_level_features, label_column='Label'):
        self.data_dir = data_dir
        self.high_level_features = high_level_features.copy()
        self.low_level_features = low_level_features.copy()
        self.label_column = label_column
        self.scaler_high = MinMaxScaler()
        self.scaler_low = MinMaxScaler()
        self.data = None
        self.mappings = {
            'category_to_id': {},
            'anomaly_to_id': {}
        }

    def load_data(self):
        """
        Loads and concatenates all CSV files from the data directory into a single DataFrame.
        """
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not csv_files:
            logging.error(f"No CSV files found in directory: {self.data_dir}")
            sys.exit(1)

        df_list = []
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                logging.info(f"Loaded {file} with shape {df.shape}")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                logging.warning(f"Loaded {file} with ISO-8859-1 encoding due to UnicodeDecodeError.")
            df_list.append(df)

        self.data = pd.concat(df_list, ignore_index=True)
        logging.info(f"Combined data shape: {self.data.shape}")

    def clean_data(self):
        """
        Cleans the dataset by handling missing values, removing duplicates, and standardizing column names.
        """
        # Standardize column names by stripping whitespace
        self.data.columns = self.data.columns.str.strip()
        logging.info("Standardized column names.")

        # Handle missing values by filling with zero
        missing_before = self.data.isnull().sum().sum()
        self.data.fillna(0, inplace=True)
        missing_after = self.data.isnull().sum().sum()
        logging.info(f"Handled missing values: before={missing_before}, after={missing_after}")

        # Remove duplicate entries
        initial_shape = self.data.shape
        self.data.drop_duplicates(inplace=True)
        final_shape = self.data.shape
        logging.info(f"Removed duplicates: {initial_shape} -> {final_shape}")

    def standardize_labels(self):
        """
        Standardizes label strings by replacing problematic characters.
        """
        if self.label_column in self.data.columns:
            self.data[self.label_column] = self.data[self.label_column].str.replace('�', '-', regex=False)
            logging.info("Standardized labels by replacing '�' with '-'.")
        else:
            logging.error(f"Label column '{self.label_column}' not found.")
            sys.exit(1)

    def map_labels(self):
        """
        Maps categorical labels to continuous integer IDs and saves the mappings.
        """
        unique_labels = self.data[self.label_column].unique()
        categories = sorted({label.split('_')[0] for label in unique_labels})

        # Map category labels to continuous integer IDs
        self.mappings['category_to_id'] = {str(category): int(idx) for idx, category in enumerate(categories)}
        # Map anomaly labels to continuous integer IDs
        sorted_unique_anomalies = sorted(unique_labels)
        self.mappings['anomaly_to_id'] = {str(label): int(idx) for idx, label in enumerate(sorted_unique_anomalies)}

        # Save mappings to JSON
        os.makedirs('./data/mappings', exist_ok=True)
        with open('./data/mappings/category_to_id.json', 'w') as f:
            json.dump(self.mappings['category_to_id'], f, indent=4)
        with open('./data/mappings/anomaly_to_id.json', 'w') as f:
            json.dump(self.mappings['anomaly_to_id'], f, indent=4)
        logging.info("Created and saved label mappings.")

        # Map labels to categories and anomalies
        self.data['Category'] = self.data[self.label_column].apply(
            lambda x: self.mappings['category_to_id'][x.split('_')[0]]
        )
        self.data['Anomaly'] = self.data[self.label_column].map(self.mappings['anomaly_to_id'])

        # Check for NaNs after initial mapping
        if self.data['Anomaly'].isnull().any():
            logging.error("Some Anomaly labels could not be mapped. Check mappings.")
            sys.exit(1)

        # Ensure Anomaly labels are continuous integers starting from 0
        unique_anomaly_labels = sorted(self.data['Anomaly'].unique())
        anomaly_continuous_id = {label: idx for idx, label in enumerate(unique_anomaly_labels)}
        self.data['Anomaly'] = self.data['Anomaly'].map(anomaly_continuous_id)

        # Verify mapping was successful
        if self.data['Anomaly'].isnull().any():
            logging.error("Some Anomaly labels could not be mapped to continuous IDs.")
            sys.exit(1)

        # Save continuous anomaly mapping
        self.mappings['anomaly_continuous_id'] = anomaly_continuous_id
        with open('./data/mappings/anomaly_continuous_id.json', 'w') as f:
            # Convert integer keys to strings for JSON compatibility
            anomaly_continuous_id_str = {str(k): v for k, v in anomaly_continuous_id.items()}
            json.dump(anomaly_continuous_id_str, f, indent=4)
        logging.info("Mapped Anomaly labels to a continuous ID range.")

    def preprocess_features(self):
        """
        Converts features to numeric, applies transformations, and normalizes.
        """
        # Convert all feature columns to numeric
        for feature in self.high_level_features + self.low_level_features:
            self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce').fillna(0)
            logging.info(f"Converted '{feature}' to numeric.")

        # Apply log1p transformation to reduce skewness
        self.apply_log_transform()
        # Clip outliers based on percentiles
        self.clip_outliers()
        # Normalize features using MinMaxScaler
        self.normalize_features()

    def apply_log_transform(self):
        """
        Applies log1p transformation to reduce skewness in specified features.
        """
        log_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s']
        for feature in log_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)
                logging.info(f"Applied log1p transformation to '{feature}'.")

    def clip_outliers(self, lower_percentile=1, upper_percentile=99):
        """
        Clips outliers in High-Level and Low-Level features based on specified percentiles.
        """
        for feature in self.high_level_features + self.low_level_features:
            lower = np.percentile(self.data[feature], lower_percentile)
            upper = np.percentile(self.data[feature], upper_percentile)
            self.data[feature] = self.data[feature].clip(lower, upper)
            logging.info(f"Clipped '{feature}' to [{lower}, {upper}].")

    def normalize_features(self):
        """
        Normalizes High-Level and Low-Level features using MinMaxScaler.
        """
        self.data[self.high_level_features] = self.scaler_high.fit_transform(self.data[self.high_level_features])
        self.data[self.low_level_features] = self.scaler_low.fit_transform(self.data[self.low_level_features])
        logging.info("Normalized features using MinMaxScaler.")

    def create_subset(self, subset_size=250, output_dir='./data/subset', classes_to_include=[0, 1, 2, 3, 4]):
        """
        Creates a balanced subset for evaluation and training.

        Args:
            subset_size (int, optional): Total number of samples in the subset. Defaults to 250.
            output_dir (str, optional): Directory to save subset files. Defaults to './data/subset'.
            classes_to_include (list, optional): List of class IDs to include. Defaults to [0, 1, 2, 3, 4].
        """
        os.makedirs(output_dir, exist_ok=True)
        samples_per_class = subset_size // len(classes_to_include)
        selected_indices = []

        for cls in classes_to_include:
            cls_indices = self.data[self.data['Category'] == cls].index.tolist()
            if len(cls_indices) < samples_per_class:
                logging.warning(f"Not enough samples for class {cls}. Available: {len(cls_indices)}, Required: {samples_per_class}")
                selected = cls_indices
            else:
                selected = np.random.choice(cls_indices, samples_per_class, replace=False).tolist()
            selected_indices.extend(selected)

        subset = self.data.loc[selected_indices].sample(frac=1).reset_index(drop=True)
        np.save(os.path.join(output_dir, 'X_high_subset.npy'), subset[self.high_level_features].values)
        np.save(os.path.join(output_dir, 'X_low_subset.npy'), subset[self.low_level_features].values)
        np.save(os.path.join(output_dir, 'y_high_subset.npy'), subset['Category'].values)
        np.save(os.path.join(output_dir, 'y_low_subset.npy'), subset['Anomaly'].values)
        logging.info(f"Saved subset with {len(subset)} samples to '{output_dir}'.")

    def split_data(self, output_dir='./data'):
        """
        Splits the full dataset into training and testing sets and saves them.

        Args:
            output_dir (str, optional): Directory to save split datasets. Defaults to './data'.
        """
        X_high = self.data[self.high_level_features].values
        X_low = self.data[self.low_level_features].values
        y_high = self.data['Category'].values
        y_low = self.data['Anomaly'].values

        X_high_train, X_high_test, X_low_train, X_low_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(
            X_high, X_low, y_high, y_low, test_size=0.2, stratify=y_high, random_state=42
        )

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'X_high_train.npy'), X_high_train)
        np.save(os.path.join(output_dir, 'X_low_train.npy'), X_low_train)
        np.save(os.path.join(output_dir, 'y_high_train.npy'), y_high_train)
        np.save(os.path.join(output_dir, 'y_low_train.npy'), y_low_train)
        np.save(os.path.join(output_dir, 'X_high_test.npy'), X_high_test)
        np.save(os.path.join(output_dir, 'X_low_test.npy'), X_low_test)
        np.save(os.path.join(output_dir, 'y_high_test.npy'), y_high_test)
        np.save(os.path.join(output_dir, 'y_low_test.npy'), y_low_test)
        logging.info("Saved training and testing datasets.")

    def preprocess(self):
        """
        Executes the full preprocessing pipeline.
        """
        self.load_data()
        self.clean_data()
        self.standardize_labels()
        self.map_labels()
        self.preprocess_features()
        self.create_subset()
        self.split_data()
        logging.info("Preprocessing pipeline completed.")


def main():
    """
    Executes data preprocessing.
    """
    high_level_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s']
    low_level_features = ['Fwd Packet Length Std', 'FIN Flag Count', 'RST Flag Count', 'Packet Length Variance']

    preprocessor = DataPreprocessor(
        data_dir='./data', 
        high_level_features=high_level_features, 
        low_level_features=low_level_features
    )
    
    try:
        preprocessor.preprocess()
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
