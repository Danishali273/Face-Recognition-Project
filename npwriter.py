import pandas as pd
import numpy as np
import os.path

# File where data will be stored
f_name = "face_encodings.csv"

def write(name, encodings):
    """
    Saves face encodings to a CSV file.
    """
    if encodings is None or len(encodings) == 0:
        return False
    
    # Check if file exists to determine if we need a header
    if os.path.isfile(f_name):
        df = pd.read_csv(f_name, index_col=0)
        
        # Create a dataframe for the new data
        # We create columns encoding_0 ... encoding_127
        latest = pd.DataFrame(encodings, columns=[f"encoding_{i}" for i in range(128)])
        latest["name"] = name
        
        df = pd.concat((df, latest), ignore_index=True, sort=False)
    else:
        # Create new file
        df = pd.DataFrame(encodings, columns=[f"encoding_{i}" for i in range(128)])
        df["name"] = name

    df.to_csv(f_name)
    return True

def get_stored_names():
    if os.path.isfile(f_name):
        try:
            df = pd.read_csv(f_name, index_col=0)
            return df["name"].unique().tolist()
        except:
            return []
    return []