import pandas as pd
import re

def load_and_preprocess_coffee_data(csv_path: str) -> pd.DataFrame:
    """
    - CSV file load
    - text columns preprocessing: lowercase, remove special characters
    """
    df = pd.read_csv(csv_path)

    # check if required columns exist
    required_columns = ["origin", "desc_1", "desc_3", "roast", "agtron"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"[에러] {col} 컬럼이 CSV에 없습니다.")
    
    # preprocess text columns
    text_cols = ["origin", "desc_1", "desc_3", "roast", "agtron"]
    for col in text_cols:
        # lowercase
        df[col] = df[col].astype(str).apply(lambda x: x.strip().lower())
        # remove special characters
        df[col] = df[col].apply(lambda x: re.sub(r'[^a-z0-9\s,/:\-\.]+', '', x))

    return df

if __name__ == "__main__":
    coffee_df = load_and_preprocess_coffee_data("../data/coffee_data.csv")
    print(coffee_df.head())
