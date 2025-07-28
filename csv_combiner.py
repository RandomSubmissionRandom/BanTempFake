import pandas as pd
from typing import List
from config import *
def combine_csv_files(file_paths: List[str], output_path: str = None) -> pd.DataFrame:
    """
    Combine multiple CSV files into one DataFrame.
    
    Args:
        file_paths: List of paths to CSV files
        output_path: Optional path to save the combined CSV
    
    Returns:
        Combined DataFrame
    """
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.dropna()
    if output_path:
        combined_df.to_csv(output_path, index=False)
    return combined_df

if __name__ == "__main__":
    combine_csv_files(["extracted_links_ht.csv", "extracted_links_gs.csv"], input_file)