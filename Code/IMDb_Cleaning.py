#Author: Yunus Khan
#Project: IMDb Movie Reviews

# Import the storage module
from google.cloud import storage
import pandas as pd
from io import StringIO
'''pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 1000)'''

# Define your bucket and folder paths
bucket_name = 'my-bigdata-project-ky'
landing_folder = f"{bucket_name}/landing/"
cleaned_folder = f"{bucket_name}/cleaned/"

# List of columns to keep
selected_columns = ["movie", "rating", "review_summary","review_detail", 
                    "review_date", "spoiler_tag","helpful_upvotes","helpful_total_votes"]

# Connect to Google Cloud Storage
storage_client = storage.Client()
blobs = storage_client.list_blobs(bucket_name, prefix="landing/")

# Process each JSON file
for blob in blobs:
    if blob.name.endswith('.json'):
        print(f"Processing {blob.name}")

        # Read the JSON file into a DataFrame
        df = pd.read_json(StringIO(blob.download_as_text()))


        # Split helpful column and drop
        df[["helpful_upvotes", "helpful_total_votes"]] = pd.DataFrame(
                df["helpful"].tolist(), 
                index=df.index)

        df.drop(columns=["helpful"], inplace=True)

        df = df[selected_columns]

        # Convert columns to appropriate data types

        df['movie'] = df['movie'].astype('string')
        df['spoiler_tag'] = df['spoiler_tag'].astype('Int64')
        df['review_summary'] = df['review_summary'].astype('string')
        df['review_detail'] = df['review_detail'].astype('string')

        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        df.dropna(subset=["review_date"], inplace=True)  # optional: remove failed dates
        df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")

        df["helpful_upvotes"] = pd.to_numeric(df["helpful_upvotes"], errors='coerce').astype('Int64')
        df["helpful_total_votes"] = pd.to_numeric(df["helpful_total_votes"], errors='coerce').astype('Int64')


        # Drop rows with any missing values
        df.dropna(inplace=True)

        # Filtering only movies as the dataset has TV shows in the movie column
        df= df[df['movie'].str.contains(r"\(\d{4}\)$", regex=True)]


        # Filter out invalid ratings
        df = df[df['rating'].between(1.0, 10.0)]

        # Remove duplicate 
        df.drop_duplicates(inplace=True)

        display(df.head(10))


        #Save the cleaned data as a Parquet file
        cleaned_filename = blob.name.replace("landing/", "cleaned/").replace(".json", ".parquet")
        df.to_parquet(f"gs://{bucket_name}/{cleaned_filename}", index=False)

        print(f"Saved cleaned file to {cleaned_filename}")
