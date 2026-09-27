data_cleaning_visualization.py
# ============================================================
# THIRANEX - DATA CLEANING & VISUALIZATION PROJECT
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ============================================================
# 2. Load Dataset
# ============================================================

file_path = "dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# 3. Basic Dataset Information
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

# ============================================================
# 4. Check Missing Values
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print("\nMissing Percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# ============================================================
# 5. Handle Missing Values
# ============================================================

for column in df.columns:

    if df[column].isnull().sum() > 0:

        # Numerical columns -> fill with median
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())

        # Categorical columns -> fill with mode
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# 6. Check Duplicate Records
# ============================================================

print("\n========== DUPLICATES ==========")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

# Remove duplicates
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# ============================================================
# 7. Statistical Summary
# ============================================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe(include="all"))

# ============================================================
# 8. Detect Outliers using IQR
# ============================================================

print("\n========== OUTLIER DETECTION ==========")

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, "-> Outliers:", len(outliers))

# ============================================================
# 9. Handle Outliers using IQR Capping
# ============================================================

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )

print("\nOutliers handled using IQR capping.")

# ============================================================
# 10. Correlation Analysis
# ============================================================

if len(numeric_columns) > 1:

    plt.figure(figsize=(10, 6))

    correlation = df[numeric_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# ============================================================
# 11. Histograms
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(df[column], bins=20, edgecolor="black")

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# ============================================================
# 12. Boxplots
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df[column])

    plt.title("Boxplot of " + column)
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()

# ============================================================
# 13. Bar Charts for Categorical Columns
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:

    # Avoid extremely large number of categories
    if df[column].nunique() <= 15:

        plt.figure(figsize=(9, 5))

        df[column].value_counts().plot(kind="bar")

        plt.title("Distribution of " + column)
        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

# ============================================================
# 14. Pair Plot
# ============================================================

if len(numeric_columns) >= 2:

    selected_columns = list(numeric_columns[:5])

    sns.pairplot(df[selected_columns])

    plt.show()

# ============================================================
# 15. Save Cleaned Dataset
# ============================================================

output_file = "cleaned_dataset.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved as:", output_file)

# ============================================================
# 16. Final Dataset Information
# ============================================================

print("\n========== FINAL DATASET ==========")

print("Final Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())

print("\n========== PROJECT COMPLETED ==========")