# scripts/preprocess.py

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
import logging
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class DataPreprocessor:
    """
    Preprocesses network intrusion detection data for hierarchical Q-Learning models.
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
        self.label_dict = {
            'category': {},
            'anomaly': {}
        }
        self.class_weights = {}

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
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                logging.info(f"Loaded {file} with shape {df.shape}")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
                logging.warning(f"Loaded {file} with ISO-8859-1 encoding due to UnicodeDecodeError.")
            # Log the columns of the current dataframe
            logging.debug(f"Columns in {file}: {df.columns.tolist()}")
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
            self.data[self.label_column] = self.data[self.label_column].astype(str).str.replace('�', '-', regex=False)
            logging.info("Standardized labels by replacing '�' with '-'.")
        else:
            logging.error(f"Label column '{self.label_column}' not found.")
            sys.exit(1)

    def map_labels(self):
        """
        Maps categorical labels to continuous integer IDs and saves the mappings.
        Also creates the label_dict for mapping IDs back to names.
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
            'Label': 'Benign',
            'Unknown': 'Unknown'
        }

        # Map labels to categories
        self.data['Category'] = self.data[self.label_column].map(lambda x: mapping.get(x, 'Unknown')).astype(str)
        unique_categories = sorted(self.data['Category'].unique())
        self.mappings['category_to_id'] = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.label_dict['category'] = {idx: cat for cat, idx in self.mappings['category_to_id'].items()}

        # Map labels to anomalies (Assuming similar mapping; adjust if different)
        self.data['Anomaly'] = self.data[self.label_column].map(lambda x: mapping.get(x, 'Unknown')).astype(str)
        unique_anomalies = sorted(self.data['Anomaly'].unique())
        self.mappings['anomaly_to_id'] = {anom: idx for idx, anom in enumerate(unique_anomalies)}
        self.label_dict['anomaly'] = {idx: anom for anom, idx in self.mappings['anomaly_to_id'].items()}

        # Save mappings to JSON
        os.makedirs('./data/mappings', exist_ok=True)
        with open('./data/mappings/category_to_id.json', 'w') as f:
            json.dump(self.mappings['category_to_id'], f, indent=4)
        with open('./data/mappings/anomaly_to_id.json', 'w') as f:
            json.dump(self.mappings['anomaly_to_id'], f, indent=4)
        logging.info("Created and saved label mappings.")

        # Save label_dict to JSON
        os.makedirs('misc', exist_ok=True)
        with open('misc/label_dict.json', 'w') as f:
            json.dump(self.label_dict, f, indent=4)
        logging.info("Created and saved label dictionary.")

        # Encode categorical labels
        self.data['Category'] = self.data['Category'].map(self.mappings['category_to_id'])
        self.data['Anomaly'] = self.data['Anomaly'].map(self.mappings['anomaly_to_id'])

        # Verify no unmapped labels
        if self.data['Category'].isnull().any() or self.data['Anomaly'].isnull().any():
            logging.error("Some labels could not be mapped. Please check the mappings.")
            sys.exit(1)

        # Convert to integer type
        self.data['Category'] = self.data['Category'].astype(int)
        self.data['Anomaly'] = self.data['Anomaly'].astype(int)

    def preprocess_features(self):
        """
        Converts features to numeric, applies transformations, and normalizes.
        """
        # Ensure all required features are present, create if missing
        for feature in self.high_level_features + self.low_level_features:
            if feature not in self.data.columns:
                logging.warning(f"Feature '{feature}' not found in data. Creating it with default value 0.")
                self.data[feature] = 0  # or another appropriate default

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
        log_features = ['Flow Duration', 'Total Fwd Pkts', 'Total Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s']
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

    def create_subset(self, subset_size=250, output_dir='./data/subset', classes_to_include=None):
        """
        Creates a balanced subset for evaluation and training.

        Args:
            subset_size (int, optional): Total number of samples in the subset. Defaults to 250.
            output_dir (str, optional): Directory to save subset files. Defaults to './data/subset'.
            classes_to_include (list, optional): List of class IDs to include. Defaults to all classes.
        """
        if classes_to_include is None:
            classes_to_include = list(self.mappings['category_to_id'].values())

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

    def compute_class_weights(self, output_dir='misc', target='category'):
        """
        Computes class weights based on the frequency of each class in the training data and saves them.

        Args:
            output_dir (str, optional): Directory to save the class weights JSON file. Defaults to 'misc'.
            target (str, optional): The target variable to compute weights for ('category' or 'anomaly'). Defaults to 'category'.
        """
        if target not in ['category', 'anomaly']:
            logging.error("Target must be either 'category' or 'anomaly'.")
            return

        y = self.data[f'{target.capitalize()}']
        class_counts = Counter(y)
        total_samples = len(y)
        num_classes = len(class_counts)

        # Compute weights as inverse frequency
        class_weights = {str(cls): total_samples / (num_classes * count) for cls, count in class_counts.items()}

        # Save to JSON
        os.makedirs(output_dir, exist_ok=True)
        class_weights_path = os.path.join(output_dir, f'{target}_class_weights.json')
        with open(class_weights_path, 'w') as f:
            json.dump(class_weights, f, indent=4)

        self.class_weights[target] = class_weights
        logging.info(f"Computed and saved class weights for '{target}' to '{class_weights_path}'.")

    def balance_data(self, df):
        """
        Balances the dataset using Random Under Sampling.

        Args:
            df (pd.DataFrame): DataFrame to balance.

        Returns:
            pd.DataFrame: Balanced DataFrame.
        """
        X = df.drop([self.label_column, "Category", "Anomaly"], axis=1, errors='ignore')
        y = df["Category"]

        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)

        df_balanced = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name='Category')], axis=1)

        # Include 'Anomaly' if it exists
        if 'Anomaly' in df.columns:
            df_balanced['Anomaly'] = df_balanced['Category'].map(lambda x: df.loc[df['Category'] == x, 'Anomaly'].iloc[0])

        logging.info(f"Balanced dataset shape: {df_balanced.shape}")

        return df_balanced

    def preprocess(self):
        """
        Executes the full preprocessing pipeline.
        """
        self.load_data()
        self.clean_data()
        self.standardize_labels()
        self.map_labels()
        self.preprocess_features()
        self.balance_and_save()
        logging.info("Preprocessing pipeline completed.")

    def balance_and_save(self):
        """
        Balances the dataset, splits into training and testing, creates subsets, computes class weights, and saves all necessary files.
        """
        # Balance the data
        balanced_df = self.balance_data(self.data)

        # Split the data
        X_high = balanced_df[self.high_level_features].values
        X_low = balanced_df[self.low_level_features].values
        y_high = balanced_df['Category'].values
        y_low = balanced_df['Anomaly'].values

        X_high_train, X_high_test, X_low_train, X_low_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(
            X_high, X_low, y_high, y_low, test_size=0.2, stratify=y_high, random_state=42
        )

        # Save full training and testing data
        np.save(os.path.join('data', 'X_high_train.npy'), X_high_train)
        np.save(os.path.join('data', 'X_low_train.npy'), X_low_train)
        np.save(os.path.join('data', 'y_high_train.npy'), y_high_train)
        np.save(os.path.join('data', 'y_low_train.npy'), y_low_train)
        np.save(os.path.join('data', 'X_high_test.npy'), X_high_test)
        np.save(os.path.join('data', 'X_low_test.npy'), X_low_test)
        np.save(os.path.join('data', 'y_high_test.npy'), y_high_test)
        np.save(os.path.join('data', 'y_low_test.npy'), y_low_test)
        logging.info("Saved full training and testing datasets.")

        # Create and save subset for overfitting
        self.create_subset(subset_size=250, output_dir='./data/subset', classes_to_include=list(self.mappings['category_to_id'].values()))
        logging.info("Created and saved subset for overfitting.")

        # Compute and save class weights
        self.compute_class_weights(output_dir='misc', target='category')
        # If needed for low-level agent, compute for 'anomaly' as well
        self.compute_class_weights(output_dir='misc', target='anomaly')

    def generate_label_dict(self):
        """
        Generates the label dictionary mapping for categories and anomalies.
        (Already handled in map_labels())
        """
        pass  # Placeholder if additional processing is needed


def main():
    """
    Executes data preprocessing.
    """
    high_level_features = ['Flow Duration', 'Total Fwd Pkts', 'Total Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s']
    low_level_features = ['Fwd Pkt Len Std', 'FIN Flag Cnt', 'RST Flag Cnt', 'Pkt Len Var']

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
