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
from sklearn.preprocessing import StandardScaler
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
        scaler_high (StandardScaler): Scaler for High-Level features.
        scaler_low (StandardScaler): Scaler for Low-Level features.
        data (pd.DataFrame): Combined and cleaned dataset.
        mappings (dict): Mappings for categories and anomalies.
    """

    def __init__(self, data_dir, high_level_features, low_level_features, label_column='Label'):
        """
        Initializes the DataPreprocessor with specified parameters.

        Args:
            data_dir (str): Directory containing raw CSV data files.
            high_level_features (list): Feature names for the High-Level Q-Network.
            low_level_features (list): Feature names for the Low-Level Q-Network.
            label_column (str, optional): Column name containing labels. Defaults to 'Label'.
        """
        self.data_dir = data_dir
        self.high_level_features = high_level_features.copy()
        self.low_level_features = low_level_features.copy()
        self.label_column = label_column
        self.scaler_high = StandardScaler()
        self.scaler_low = StandardScaler()
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
        Maps categorical labels to integer IDs and separates them into categories and anomalies.
        """
        unique_labels = self.data[self.label_column].unique()
        categories = sorted({label.split('_')[0] for label in unique_labels})
        self.mappings['category_to_id'] = {category: idx for idx, category in enumerate(categories)}
        self.mappings['anomaly_to_id'] = {label: idx for idx, label in enumerate(unique_labels)}

        # Save mappings
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

        # Remove entries with unmapped labels
        initial_shape = self.data.shape
        self.data.dropna(subset=['Category', 'Anomaly'], inplace=True)
        final_shape = self.data.shape
        if initial_shape != final_shape:
            removed = initial_shape[0] - final_shape[0]
            logging.warning(f"Removed {removed} entries with unmapped labels.")

    def convert_features(self):
        """
        Ensures all feature columns are numeric and handles conversion errors.
        """
        for feature in self.high_level_features + self.low_level_features:
            if not pd.api.types.is_numeric_dtype(self.data[feature]):
                self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce').fillna(0)
                logging.info(f"Converted feature '{feature}' to numeric.")

    def handle_negatives(self):
        """
        Sets negative values in High-Level features to zero.
        """
        for feature in self.high_level_features:
            negatives = (self.data[feature] < 0).sum()
            if negatives > 0:
                self.data.loc[self.data[feature] < 0, feature] = 0
                logging.info(f"Set {negatives} negative values in '{feature}' to 0.")

    def apply_log_transform(self):
        """
        Applies log1p transformation to selected features to handle skewness.
        """
        log_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Flow Bytes/s', 'Flow Packets/s']
        for feature in log_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)
                logging.info(f"Applied log1p transformation to '{feature}'.")

    def clip_outliers(self, lower_percentile=1, upper_percentile=99):
        """
        Clips outliers in High-Level and Low-Level features based on specified percentiles.

        Args:
            lower_percentile (int, optional): Lower percentile for clipping. Defaults to 1.
            upper_percentile (int, optional): Upper percentile for clipping. Defaults to 99.
        """
        for feature in self.high_level_features + self.low_level_features:
            if feature in self.data.columns:
                lower = np.percentile(self.data[feature], lower_percentile)
                upper = np.percentile(self.data[feature], upper_percentile)
                self.data[feature] = self.data[feature].clip(lower, upper)
                logging.info(f"Clipped '{feature}' to [{lower}, {upper}].")

    def normalize_features(self):
        """
        Normalizes High-Level and Low-Level features using StandardScaler.
        """
        self.data[self.high_level_features] = self.scaler_high.fit_transform(self.data[self.high_level_features])
        self.data[self.low_level_features] = self.scaler_low.fit_transform(self.data[self.low_level_features])
        logging.info("Normalized High-Level and Low-Level features.")

    def remove_uninformative_features(self):
        """
        Removes features with a single unique value as they are uninformative and updates feature lists.
        """
        features_to_remove = []
        for feature in self.high_level_features + self.low_level_features:
            unique_values = self.data[feature].nunique()
            if unique_values <= 1:
                features_to_remove.append(feature)
                logging.info(f"Marked '{feature}' for removal with {unique_values} unique value(s).")

        if features_to_remove:
            self.data.drop(columns=features_to_remove, inplace=True)
            logging.info(f"Removed uninformative features: {features_to_remove}")

            # Remove from feature lists
            self.high_level_features = [f for f in self.high_level_features if f not in features_to_remove]
            self.low_level_features = [f for f in self.low_level_features if f not in features_to_remove]
            logging.info(f"Updated High-Level Features: {self.high_level_features}")
            logging.info(f"Updated Low-Level Features: {self.low_level_features}")
        else:
            logging.info("No uninformative features found.")

    def create_subset(self, subset_size=250, output_dir='./data/subset', classes_to_include=[0, 1, 2, 3, 4]):
        """
        Creates and saves a subset of the data for training and evaluation.

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
        X_high_subset = subset[self.high_level_features].values
        X_low_subset = subset[self.low_level_features].values
        y_high_subset = subset['Category'].values
        y_low_subset = subset['Anomaly'].values

        # Save subsets
        np.save(os.path.join(output_dir, 'X_high_subset.npy'), X_high_subset)
        np.save(os.path.join(output_dir, 'X_low_subset.npy'), X_low_subset)
        np.save(os.path.join(output_dir, 'y_high_subset.npy'), y_high_subset)
        np.save(os.path.join(output_dir, 'y_low_subset.npy'), y_low_subset)

        # Summary
        logging.info(f"Created subset with {len(subset)} samples and saved to '{output_dir}'.")
        for cls in classes_to_include:
            count = np.sum(y_high_subset == cls)
            label_name = [k for k, v in self.mappings['category_to_id'].items() if v == cls]
            label_name = label_name[0] if label_name else "Unknown"
            logging.info(f"- {count} samples for class '{label_name}' (Label ID {cls})")

    def preprocess(self):
        """
        Executes the full preprocessing pipeline.

        Returns:
            pd.DataFrame: The fully preprocessed dataset.
        """
        self.load_data()
        self.clean_data()
        self.standardize_labels()
        self.map_labels()
        self.convert_features()
        self.handle_negatives()
        self.apply_log_transform()
        self.clip_outliers()
        self.normalize_features()
        self.remove_uninformative_features()
        logging.info("Completed preprocessing pipeline.")
        return self.data


def main():
    """
    Main function to execute data preprocessing.
    """
    # Define feature lists
    high_level_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s'
    ]
    low_level_features = [
        'Fwd Packet Length Std', 'FIN Flag Count', 'RST Flag Count',
        'Packet Length Variance'
    ]

    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        data_dir='./data',  # Adjust path as needed
        high_level_features=high_level_features,
        low_level_features=low_level_features,
        label_column='Label'
    )

    try:
        # Execute preprocessing
        data = preprocessor.preprocess()
    except KeyError as e:
        logging.error(f"KeyError: {e}. Check if all feature columns are present.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during preprocessing: {e}")
        sys.exit(1)

    # Create subset
    preprocessor.create_subset(subset_size=250, output_dir='./data/subset', classes_to_include=[0, 1, 2, 3, 4])

    # Split data into training and testing sets
    X_high = data[preprocessor.high_level_features].values
    X_low = data[preprocessor.low_level_features].values
    y = data['Category'].values

    X_high_train, X_high_test, X_low_train, X_low_test, y_train, y_test = train_test_split(
        X_high, X_low, y, test_size=0.2, stratify=y, random_state=42
    )
    logging.info(f"Split data into training and testing sets.")

    # Save training and testing datasets
    np.save('./data/X_high_train.npy', X_high_train)
    np.save('./data/X_low_train.npy', X_low_train)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/X_high_test.npy', X_high_test)
    np.save('./data/X_low_test.npy', X_low_test)
    np.save('./data/y_test.npy', y_test)
    logging.info("Saved training and testing datasets.")

    # Save mappings (already saved during mapping)
    logging.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
