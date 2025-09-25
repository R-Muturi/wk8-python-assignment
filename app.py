import pandas as pd
import numpy as np

# Load the metadata file into a DataFrame
try:
    df = pd.read_csv('metadata.csv', low_memory=False)
except FileNotFoundError:
    print("Please make sure 'metadata.csv' is in the same directory.")
    exit()

# Examine the data structure and dimensions
print("DataFrame Shape:", df.shape)
print("\nFirst 5 rows of the DataFrame:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# Create a copy to preserve the original DataFrame
df_cleaned = df.copy()

# Handle missing values in key columns
# We'll drop rows where 'title' or 'abstract' are missing
df_cleaned.dropna(subset=['title', 'abstract', 'publish_time'], inplace=True)

# Convert 'publish_time' to datetime format
df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')

# Drop any rows where the conversion failed (NaT)
df_cleaned.dropna(subset=['publish_time'], inplace=True)

# Extract the year for time-based analysis
df_cleaned['publish_year'] = df_cleaned['publish_time'].dt.year

print("\nShape of cleaned DataFrame:", df_cleaned.shape)
print("\nData types after cleaning:")
print(df_cleaned.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns

# Analysis 1: Publications over time
publications_by_year = df_cleaned['publish_year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
publications_by_year.plot(kind='line', marker='o')
plt.title('Number of Publications Over Time')
plt.xlabel('Publication Year')
plt.ylabel('Number of Papers')
plt.grid(True)
plt.show()

# Analysis 2: Top 10 journals
top_journals = df_cleaned['journal'].value_counts().head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_journals.values, y=top_journals.index, palette='viridis')
plt.title('Top 10 Journals by Publication Count')
plt.xlabel('Number of Papers')
plt.ylabel('Journal')
plt.show()

from collections import Counter
import re

# Simple function to get word frequencies
def get_top_words(text_series, n=20):
    all_words = ' '.join(text_series.dropna()).lower()
    # Remove punctuation and numbers
    all_words = re.sub(r'[^a-z\s]', '', all_words)
    words = all_words.split()
    # Filter out common stopwords
    stopwords = set(plt.matplotlib.font_manager.get_font_names()) # A simple way to get some stopwords
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    return Counter(filtered_words).most_common(n)

top_title_words = get_top_words(df_cleaned['title'])
print("\nTop 20 words in paper titles:")
print(top_title_words)

