# ================================
# 8. Visualization of SVM decision boundary in PCA 2D
# ================================

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Necessary import for SVM
import matplotlib.pyplot as plt
import os

# ===============================
# 1. Read and clean Excel or CSV file
# ===============================

# Set the input file name (change as needed)
excel_file = "SPM_data.xlsm"

# Get the file extension to determine how to read the file
file_ext = os.path.splitext(excel_file)[1].lower()

if file_ext in [".csv", ".txt"]:
    # If the file is a CSV or TXT, read it as a CSV with comma delimiter
    df_raw = pd.read_csv(excel_file, delimiter=",", header=None)
else:
    # If the file is an Excel file, read it using read_excel
    df_temp = pd.read_excel(excel_file, header=None)
    # If the data is in a single column (comma-separated), split it into columns
    if df_temp.shape[1] == 1:
        df_raw = df_temp[0].str.split(",", expand=True)
    else:
        # Otherwise, use the DataFrame as is
        df_raw = df_temp

# Print the raw DataFrame to inspect the data
print(df_raw)

# Save the cleaned DataFrame to a new Excel file for easy opening in Excel
excel_file_limpio = "Data_limpio.xlsx"
df_raw.to_excel(excel_file_limpio, index=False, header=False)
print(f"Cleaned file saved as: {excel_file_limpio}")
