"""
Data exploration script for Census dataset.

This script:
1. Loads the census.csv data
2. Checks for "messy" data (whitespace issues)
3. Generates a comprehensive profiling report using ydata-profiling
4. Saves the report as HTML
"""

import os
import sys

import pandas as pd
from ydata_profiling import ProfileReport


def check_messy_data(df):
    """
    Check for whitespace issues in column names and values.

    Args:
        df: pandas DataFrame to check

    Returns:
        list: List of issues found
    """
    issues = []

    # Check column names for whitespace
    print("\n" + "=" * 60)
    print("Checking for 'messy' data (whitespace issues)...")
    print("=" * 60)

    # Check column names
    for col in df.columns:
        original_col = col
        stripped_col = col.strip()

        if original_col != stripped_col:
            issues.append(f"Column name '{original_col}' has whitespace")
            print(f"  ⚠️  Column '{original_col}' has whitespace")

        if col.startswith(" ") or col.endswith(" "):
            msg = f"Column '{original_col}' has leading/trailing whitespace"
            issues.append(msg)
            print(f"  ⚠️  {msg}")

    # Check string values for leading/trailing whitespace
    print("\nChecking string values for whitespace...")
    string_columns = df.select_dtypes(include=['object']).columns
    whitespace_found = False

    for col in string_columns:
        # Check first few rows for whitespace
        sample = df[col].head(10)
        for idx, value in sample.items():
            if pd.notna(value) and str(value) != str(value).strip():
                if not whitespace_found:
                    msg = "  ⚠️  Found whitespace in values (showing first):"
                    print(msg)
                    whitespace_found = True
                print(f"      Column '{col}', Row {idx}: '{value}'")
                issues.append(f"Value in column '{col}' has whitespace")
                break  # Only show first occurrence per column

    if not issues and not whitespace_found:
        print("  ✓ No whitespace issues found!")
    else:
        print(f"\n  Found {len(issues)} potential issues")
        print("  💡 Tip: Clean the data by removing spaces before training")

    return issues


def main():
    """Main function to explore the census data."""

    # File path
    data_file = "data/census.csv"

    # Check if file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: File '{data_file}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        print("   Please make sure you're in the project directory")
        sys.exit(1)

    print("=" * 60)
    print("Census Data Exploration")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading data from: {data_file}")
    try:
        df = pd.read_csv(data_file)
        print("  ✓ Data loaded successfully!")
        print(f"  ✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        sys.exit(1)

    # Show basic info
    print("\n" + "=" * 60)
    print("Basic Data Information")
    print("=" * 60)
    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst few rows:")
    print(df.head())

    # Check for messy data
    issues = check_messy_data(df)

    # Generate profiling report
    print("\n" + "=" * 60)
    print("Generating Profiling Report")
    print("=" * 60)
    print("\n⏳ This may take a few minutes for large datasets...")

    try:
        # Create profile report
        profile = ProfileReport(
            df,
            title="Census Income Dataset - Profiling Report",
            dataset={
                "description": (
                    "Census income prediction dataset from "
                    "UCI ML Repository"
                ),
                "url": "https://archive.ics.uci.edu/ml/datasets/census+income"
            },
            variables={
                "descriptions": {
                    "age": "Age of the individual",
                    "workclass": "Type of employment",
                    "fnlgt": "Final weight (demographic weighting)",
                    "education": "Highest level of education",
                    "education-num": "Numeric representation of education",
                    "marital-status": "Marital status",
                    "occupation": "Type of occupation",
                    "relationship": "Relationship status",
                    "race": "Race",
                    "sex": "Gender",
                    "capital-gain": "Capital gains",
                    "capital-loss": "Capital losses",
                    "hours-per-week": "Hours worked per week",
                    "native-country": "Country of origin",
                    "salary": "Income level (target variable)"
                }
            },
            minimal=False  # Full report
        )

        # Save report
        output_file = "census_data_report.html"
        print(f"\n💾 Saving report to: {output_file}")
        profile.to_file(output_file)

        print("\n" + "=" * 60)
        print("✅ Report Generated Successfully!")
        print("=" * 60)
        print(f"\n📊 Report saved as: {output_file}")
        print("   Open this file in your browser to view the analysis")
        print("\n   The report includes:")
        print("   • Overview statistics")
        print("   • Variable distributions")
        print("   • Correlations")
        print("   • Missing values analysis")
        print("   • Sample data")

        if issues:
            print(f"\n⚠️  Note: {len(issues)} data quality issues found")
            print("   Consider cleaning the data before training")

    except ImportError:
        print("\n❌ Error: ydata-profiling is not installed!")
        print("   Install it with: pip install ydata-profiling")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
