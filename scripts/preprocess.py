# scripts/preprocess.py

import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sys

class DataPreprocessor:
    def __init__(self, data_dir, high_level_features, low_level_features, label_column='Label'):
        self.data_dir = data_dir
        self.high_level_features = high_level_features
        self.low_level_features = low_level_features
        self.label_column = label_column
        self.scaler_high = StandardScaler()
        self.scaler_low = StandardScaler()
        self.data = None
        self.X_high = None
        self.X_low = None
        self.y_category = None
        self.y_anomaly = None
        self.label_to_category = {}
        self.label_to_anomaly = {}
        self.category_to_id = {}
        self.anomaly_to_id = {}

    def load_data(self):
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not all_files:
            print(f"No CSV files found in directory: {self.data_dir}")
            sys.exit(1)
        df_list = []
        for file in all_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to a different encoding if utf-8 fails
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                print(f"Loaded {file} with ISO-8859-1 encoding due to UnicodeDecodeError.")
            df_list.append(df)
            print(f"Loaded {file} with shape {df.shape}")
        self.data = pd.concat(df_list, ignore_index=True)
        print(f"Combined data shape: {self.data.shape}")

    def clean_data(self):
        # Handle missing values
        missing_before = self.data.isnull().sum().sum()
        self.data.fillna(0, inplace=True)
        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values before: {missing_before}, after filling: {missing_after}")
        # Remove duplicates
        initial_shape = self.data.shape
        self.data.drop_duplicates(inplace=True)
        print(f"Data cleaned: Removed duplicates {initial_shape} -> {self.data.shape}")

    def clean_column_names(self):
        # Strip leading/trailing spaces and standardize column names
        self.data.columns = [col.strip() for col in self.data.columns]
        print(f"Cleaned column names: {self.data.columns.tolist()}")

    def extract_unique_labels(self):
        # Check if the label column exists
        if self.label_column not in self.data.columns:
            raise KeyError(f"'{self.label_column}' column not found in the dataset. Available columns: {self.data.columns.tolist()}")
        
        unique_labels = self.data[self.label_column].unique()
        print(f"Unique labels found ({len(unique_labels)}): {unique_labels}")
        return unique_labels

    def extract_categories(self, unique_labels):
        categories = set()
        for label in unique_labels:
            # Extract category by splitting the label string
            if '_' in label:
                category = label.split('_')[0]
            else:
                category = label
            categories.add(category)
        print(f"Unique categories extracted ({len(categories)}): {sorted(categories)}")
        return sorted(list(categories))

    def create_mappings(self, unique_labels, categories):
        # Map categories to integers
        self.category_to_id = {category: idx for idx, category in enumerate(categories)}
        print(f"Category to ID mapping: {self.category_to_id}")
        
        # Map labels to integers
        self.anomaly_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Anomaly to ID mapping: {self.anomaly_to_id}")
        
        # Map labels to categories (as integers)
        for label in unique_labels:
            if '_' in label:
                category = label.split('_')[0]
            else:
                category = label
            category_id = self.category_to_id.get(category, -1)
            self.label_to_category[label] = category_id
            self.label_to_anomaly[label] = self.anomaly_to_id[label]
        print("Label to Category and Anomaly mappings created.")

    def map_labels(self):
        # Apply mappings to the data
        self.data['Category'] = self.data[self.label_column].map(self.label_to_category)
        self.data['Anomaly'] = self.data[self.label_column].map(self.label_to_anomaly)
        
        # Remove any entries with unknown categories
        num_unknown = self.data['Category'].isna().sum()
        if num_unknown > 0:
            print(f"Warning: {num_unknown} instances have unknown categories and will be removed.")
            self.data = self.data.dropna(subset=['Category', 'Anomaly'])
        
        self.y_category = self.data['Category'].astype(int).values
        self.y_anomaly = self.data['Anomaly'].astype(int).values

    def standardize_labels(self):
        print("\nStandardizing Labels:")
        # Replace '�' with '-' or another appropriate character
        self.data[self.label_column] = self.data[self.label_column].str.replace('�', '-', regex=False)
        print("Replaced '�' with '-' in labels.")

    def remove_duplicate_columns(self):
        print("\nRemoving Duplicate Columns:")
        duplicated_columns = self.data.columns[self.data.columns.duplicated()]
        if duplicated_columns.any():
            self.data.drop(columns=duplicated_columns, inplace=True)
            print(f"Removed duplicated columns: {duplicated_columns.tolist()}")
        else:
            print("No duplicated columns found.")

    def select_features(self):
        # Ensure features exist
        missing_high = [feat for feat in self.high_level_features if feat not in self.data.columns]
        missing_low = [feat for feat in self.low_level_features if feat not in self.data.columns]
        if missing_high or missing_low:
            raise ValueError(f"Missing features: {missing_high + missing_low}")
        self.X_high = self.data[self.high_level_features].values.astype(float)
        self.X_low = self.data[self.low_level_features].values.astype(float)
        print("Features selected for high-level and low-level agents.")

    def convert_to_numeric(self):
        print("\nConverting Features to Numeric:")
        for feature in self.high_level_features + self.low_level_features:
            if not pd.api.types.is_numeric_dtype(self.data[feature]):
                self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce').fillna(0)
                print(f"Converted '{feature}' to numeric.")

    def handle_negative_values(self):
        print("\nHandling Negative Values:")
        for feature in self.high_level_features:
            if feature in self.data.columns:
                num_neg = (self.data[feature] < 0).sum()
                if num_neg > 0:
                    self.data.loc[self.data[feature] < 0, feature] = 0
                    print(f"Set {num_neg} negative values in '{feature}' to 0.")

    def apply_log_transform(self):
        # Apply log1p (log(1 + x)) to handle zero values and reduce skewness
        log_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Flow Bytes/s', 'Flow Packets/s']
        for feature in log_features:
            if feature in self.data.columns:
                # To prevent taking log of zero, ensure all values are >=0
                self.data[feature] = self.data[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)
                print(f"Applied log1p transformation to '{feature}'")

    def inspect_features(self):
        print("\nInspecting Features for Anomalies:")
        for idx, feature in enumerate(self.high_level_features):
            col = self.X_high[:, idx]
            num_inf = np.isinf(col).sum()
            num_nan = np.isnan(col).sum()
            max_val = np.max(col)
            min_val = np.min(col)
            print(f"High-Level Feature '{feature}': inf={num_inf}, NaN={num_nan}, min={min_val}, max={max_val}")

        for idx, feature in enumerate(self.low_level_features):
            col = self.X_low[:, idx]
            num_inf = np.isinf(col).sum()
            num_nan = np.isnan(col).sum()
            max_val = np.max(col)
            min_val = np.min(col)
            print(f"Low-Level Feature '{feature}': inf={num_inf}, NaN={num_nan}, min={min_val}, max={max_val}")

    def clip_outliers(self, lower_percentile=1, upper_percentile=99):
        print("\nClipping Outliers:")
        for idx, feature in enumerate(self.high_level_features):
            col = self.X_high[:, idx]
            lower = np.percentile(col, lower_percentile)
            upper = np.percentile(col, upper_percentile)
            self.X_high[:, idx] = np.clip(col, lower, upper)
            print(f"High-Level Feature '{feature}': clipped to [{lower}, {upper}]")

        for idx, feature in enumerate(self.low_level_features):
            col = self.X_low[:, idx]
            lower = np.percentile(col, lower_percentile)
            upper = np.percentile(col, upper_percentile)
            self.X_low[:, idx] = np.clip(col, lower, upper)
            print(f"Low-Level Feature '{feature}': clipped to [{lower}, {upper}]")

    def replace_inf_values(self):
        print("\nReplacing Infinity Values:")
        self.X_high[np.isinf(self.X_high)] = np.nan
        self.X_low[np.isinf(self.X_low)] = np.nan
        # Optionally, replace NaN with a large finite value or the maximum value in the feature
        self.X_high = np.nan_to_num(self.X_high, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_low = np.nan_to_num(self.X_low, nan=0.0, posinf=0.0, neginf=0.0)
        print("Replaced 'inf' values with 0.0.")

    def inspect_rst_flag_count(self):
        print("\nInspecting 'RST Flag Count':")
        if 'RST Flag Count' in self.data.columns:
            unique_values = self.data['RST Flag Count'].unique()
            print(f"Unique values in 'RST Flag Count': {unique_values}")
        else:
            print("'RST Flag Count' feature not found.")

    def exclude_uninformative_features(self):
        print("\nExcluding Uninformative Features:")
        # Example: Exclude 'RST Flag Count' if all values are 0
        if 'RST Flag Count' in self.high_level_features:
            unique_vals = self.data['RST Flag Count'].unique()
            if len(unique_vals) == 1 and unique_vals[0] == 0.0:
                self.high_level_features.remove('RST Flag Count')
                print("Excluded 'RST Flag Count' from high-level features.")
        if 'RST Flag Count' in self.low_level_features:
            unique_vals = self.data['RST Flag Count'].unique()
            if len(unique_vals) == 1 and unique_vals[0] == 0.0:
                self.low_level_features.remove('RST Flag Count')
                print("Excluded 'RST Flag Count' from low-level features.")

    def normalize_features(self):
        print("\nNormalizing Features:")
        try:
            self.X_high = self.scaler_high.fit_transform(self.X_high)
            print("High-Level Features normalized.")
        except Exception as e:
            raise ValueError(f"Error during normalization of high-level features: {e}")
        
        try:
            self.X_low = self.scaler_low.fit_transform(self.X_low)
            print("Low-Level Features normalized.")
        except Exception as e:
            raise ValueError(f"Error during normalization of low-level features: {e}")

    def split_data(self, test_size=0.2, random_state=42):
        X_high_train, X_high_test, X_low_train, X_low_test, y_train, y_test = train_test_split(
            self.X_high, self.X_low, self.y_category, test_size=test_size, stratify=self.y_category, random_state=random_state)
        print(f"Data split into train and test sets with test size {test_size}")
        return (X_high_train, X_low_train, y_train), (X_high_test, X_low_test, y_test)

    def save_mappings(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'category_to_id.json'), 'w') as f:
            json.dump(self.category_to_id, f, indent=4)
        with open(os.path.join(output_dir, 'anomaly_to_id.json'), 'w') as f:
            json.dump(self.anomaly_to_id, f, indent=4)
        print("Mappings saved to JSON files.")

    def preprocess(self):
        self.load_data()
        self.clean_column_names()
        self.clean_data()
        unique_labels = self.extract_unique_labels()
        categories = self.extract_categories(unique_labels)
        self.create_mappings(unique_labels, categories)
        self.map_labels()
        self.standardize_labels()
        self.remove_duplicate_columns()
        self.handle_negative_values()
        self.apply_log_transform()
        self.select_features()
        self.convert_to_numeric()
        self.inspect_features()
        self.clip_outliers()
        self.replace_inf_values()
        self.inspect_rst_flag_count()
        self.exclude_uninformative_features()
        self.normalize_features()
        return self.split_data()

def main():
    # Define feature lists based on the sample header
    high_level_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s'
    ]
    low_level_features = [
        'Fwd Packet Length Std', 'FIN Flag Count', 'RST Flag Count',
        'Packet Length Variance'
    ]
    
    # Initialize DataPreprocessor
    data_dir = './data'  # Adjust path if necessary
    preprocessor = DataPreprocessor(
        data_dir=data_dir,
        high_level_features=high_level_features,
        low_level_features=low_level_features,
        label_column='Label'
    )
    
    try:
        # Preprocess data
        (X_high_train, X_low_train, y_train), (X_high_test, X_low_test, y_test) = preprocessor.preprocess()
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        print("Please verify that the 'Label' column exists in your dataset.")
        print(f"Available columns: {preprocessor.data.columns.tolist()}")
        sys.exit(1)
    except ValueError as e:
        print(f"ValueError encountered: {e}")
        print("This might be due to infinite or excessively large values in the data.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        sys.exit(1)
    
    # Save mappings
    preprocessor.save_mappings(output_dir='./data/mappings')
    
    # Save preprocessed data as numpy arrays
    np.save('./data/X_high_train.npy', X_high_train)
    np.save('./data/X_low_train.npy', X_low_train)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/X_high_test.npy', X_high_test)
    np.save('./data/X_low_test.npy', X_low_test)
    np.save('./data/y_test.npy', y_test)
    print("Preprocessed data saved successfully.")

if __name__ == "__main__":
    main()
