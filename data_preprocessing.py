# src/data_preprocessing.py

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the dataset into a DataFrame and replace empty strings with NaN.
    """
    df = pd.read_csv(file_path)
    df.replace("", np.nan, inplace=True)
    return df

def inspect_data(df):
    """
    Inspect the dataset for data types, missing values, and duplicates.
    """
    print("Data Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicates:")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

def handle_missing_values(df):
    """
    Handle missing values in the dataset:
    - Numerical columns: Impute with mean.
    - Categorical columns: Impute with mode or 'Unknown'.
    - Date columns: Forward-fill missing values.
    """
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Impute numerical columns with mean
            df[col].fillna(df[col].mean(), inplace=True)
        elif df[col].dtype == 'object':
            if "Date" in col:
                # Forward-fill missing date values
                df[col].fillna(method='ffill', inplace=True)
            else:
                # Impute categorical columns with mode
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    """
    return df.drop_duplicates()

def preprocess_data(file_path, save_path):
    """
    Load, clean, and preprocess the dataset.
    """
    print("Loading data...")
    df = load_data(file_path)
    
    print("Inspecting data...")
    inspect_data(df)
    
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    
    print("\nRemoving duplicates...")
    df = remove_duplicates(df)
    
    # Save the cleaned dataset
    print(f"\nSaving cleaned data to {save_path}...")
    df.to_csv(save_path, index=False)
    print("Data preprocessing complete!")
    return df

# For testing purposes (run this script directly to clean your data)
if __name__ == "__main__":
    raw_file_path = "../data/raw/shipment_data.csv"
    processed_file_path = "../data/processed/shipment_data_cleaned.csv"
    
    preprocess_data(raw_file_path, processed_file_path)
