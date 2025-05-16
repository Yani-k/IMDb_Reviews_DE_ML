#Author: Yunus
#Data Visualization and Feature Importance

# This notebook includes visualizations and feature importance analysis for the IMDb movie review prediction model.

# ------------------
# Imports
# ------------------
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage
import os

def upload_plot_to_gcs(local_path, gcs_path):
    client = storage.Client()
    bucket = client.bucket('my-bigdata-project-ky')
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    os.remove(local_path)  # Clean up local copy

# ------------------
# Setup
# ------------------
sns.set(style="whitegrid")
spark = SparkSession.builder.getOrCreate()

# Load Spark predictions and convert to pandas
predictions = spark.read.parquet("gs://my-bigdata-project-ky/trusted/predictions.parquet")
predictions_pd = predictions.select("rating", "prediction", "label", "helpful_upvotes", "helpful_total_votes").toPandas()

# Load cleaned data and select relevant columns
cleaned_df = spark.read.parquet("gs://my-bigdata-project-ky/cleaned/")
cleaned_df = cleaned_df.select("rating", "helpful_upvotes", "helpful_total_votes").toPandas()

# Load CrossValidatorModel and extract best Logistic Regression stage
cv_model = CrossValidatorModel.load("gs://my-bigdata-project-ky/models/LogisticRegression_model")
lr_model = cv_model.bestModel.stages[-1]  # Last stage in pipeline is Logistic Regression

# Define input feature names manually (used in training)
# Extract text feature importance (only one vector feature â€” overall weight)
all_coefficients = lr_model.coefficients.toArray()

text_feature_importance = pd.DataFrame({
    "feature": ["text_features"],
    "coefficient": [all_coefficients[0]]
})

# Define numeric feature names and their coefficients
# Extract numeric feature coefficients
numeric_feature_names = ["helpful_upvotes", "helpful_total_votes", "review_detail_wordcount"] 
numeric_coefficients = all_coefficients[-3:]
numeric_feature_importance = pd.DataFrame({
    "feature": numeric_feature_names,
    "coefficient": numeric_coefficients
})

importance_df = pd.concat([text_feature_importance, numeric_feature_importance])
importance_df["category"] = importance_df["feature"].apply(lambda x: "text" if x == "text_features" else "numeric")
importance_df = importance_df.sort_values(by="coefficient", ascending=False)




# 1. Prediction Class Count
# This bar chart provides a simple overview of how the model's predictions are distributed.
# It's useful to quickly check if the model is biased towards a class or is balanced in sentiment prediction.

plt.figure()
sns.countplot(x="prediction", data=predictions_pd)
plt.title("Count of Predicted Sentiments")
plt.xlabel("Predicted Class (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
local_path = "prediction_count.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/prediction_count.png")
plt.show()


# 2. Average Rating by Predicted Sentiment
# This bar plot compares average actual ratings for each predicted class.
# It's a sanity check to ensure predictions align with real-world user ratings.
plt.figure()
avg_rating_pred = predictions_pd.groupby("prediction")["rating"].mean().reset_index()
sns.barplot(x="prediction", y="rating", data=avg_rating_pred)
plt.title("Average Rating by Predicted Sentiment")
plt.xlabel("Predicted Class (0 = Negative, 1 = Positive)")
plt.ylabel("Average Rating")
local_path = "avg_rating_by_prediction.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/avg_rating_by_prediction.png")
plt.show()

# 3. Feature Importance Bar Chart
# Visualizes feature weights learned by the model to interpret variable influence.

importance_df_rounded = importance_df.copy()
importance_df_rounded["coefficient"] = importance_df_rounded["coefficient"].round(4)
importance_df_rounded = importance_df_rounded.sort_values(by="coefficient", ascending=True).reset_index(drop=True)

plt.figure(figsize=(8, 4))
ax = sns.barplot(y="coefficient", x="feature", data=importance_df_rounded, hue="category", dodge=False)
ax.invert_xaxis()  # Reverse axis to highlight most important feature

# Add coefficient values inside bars
for i, coef in enumerate(importance_df_rounded["coefficient"]):
    ax.text(i, coef / 2, f"{coef:.4f}", ha='center', va='center', fontsize=9, color='red')

plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.legend(title="Category")
plt.ylim(-0.3, 0.3)
plt.tight_layout()
local_path = "feature_importance.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/feature_importance.png")
plt.show()


# 4. Top 10 Most Reviewed Movies (with Count Labels)
# A horizontal bar chart highlighting the most reviewed movie titles.
# This shows which titles attracted the most attention and may reflect popularity or controversy.

sdf_top_movies = spark.read.parquet("gs://my-bigdata-project-ky/cleaned/")
sdf_top_movies_pd = sdf_top_movies.select("movie").toPandas()
top_movies = sdf_top_movies_pd["movie"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_movies.values, y=top_movies.index, palette="crest")
plt.title("Top 10 Most Reviewed Movies")
plt.xlabel("Number of Reviews")
plt.ylabel("Movie Title")
for i, v in enumerate(top_movies.values):
    plt.text(v + 1, i, str(v), color='black', va='center')
plt.tight_layout()
local_path = "top10_reviewed_movies.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/top10_reviewed_movies.png")
plt.show()

# 5. Average Rating by Review Month (Trend)
# This line chart reveals trends in viewer sentiment over time.
# Useful for observing whether ratings improve or decline during certain periods.
sdf_monthly = spark.read.parquet("gs://my-bigdata-project-ky/trusted/feature_engineered_data.parquet")
sdf_monthly_pd = sdf_monthly.select("review_yearmonth", "rating").toPandas()
monthly_avg = sdf_monthly_pd.groupby("review_yearmonth")["rating"].mean().sort_index()
plt.figure(figsize=(12, 4))
monthly_avg.plot(marker='o')
plt.title("Average Rating Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Average Rating")
local_path = "rating_trend_over_time.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/rating_trend_over_time.png")
plt.show()

# 6. Confusion Matrix
from sklearn.metrics import confusion_matrix
import numpy as np
conf_mat = confusion_matrix(predictions_pd['label'], predictions_pd['prediction'])
plt.figure(figsize=(4, 3))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
local_path = "confusion_matrix.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/confusion_matrix.png")
plt.show()

# 7. ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
preds_pd = predictions.select("label", "probability").toPandas()
preds_pd["score"] = preds_pd["probability"].apply(lambda x: float(x[1]))
fpr, tpr, thresholds = roc_curve(preds_pd["label"], preds_pd["score"])
roc_auc = roc_auc_score(preds_pd["label"], preds_pd["score"])
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
local_path = "roc_curve.png"
plt.savefig(local_path)
upload_plot_to_gcs(local_path, "visualisations/roc_curve.png")
plt.show()
