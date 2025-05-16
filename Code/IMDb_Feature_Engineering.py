

# Author: Yunus Khan


# ----------------------------------------
# Import required PySpark ML modules
# ----------------------------------------
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import to_date, year, month, length, size, split, col, when
from pyspark.ml import Pipeline

# ----------------------------------------
# Load cleaned data from cleaned/ folder
# ----------------------------------------
sdf = spark.read.parquet("gs://my-bigdata-project-ky/cleaned/")

# ----------------------------------------
# Feature Engineering
# ----------------------------------------

# Convert string date to proper date type
sdf = sdf.withColumn("review_date", to_date(col("review_date"), "yyyy-MM-dd"))

# Filter nulls and apply custom logic
sdf = sdf.filter((col("rating").isNotNull()) & (col("review_detail").isNotNull()) & (col("review_summary").isNotNull()))
sdf = sdf.withColumn("review_detail_wordcount", size(split(col("review_detail"), " ")))
sdf = sdf.filter(length(col("review_detail")) > 10)
sdf = sdf.filter(col("review_detail_wordcount") > 5)

# Create year/month/yearmonth columns
sdf = sdf.withColumn("review_year", year(col("review_date")))
sdf = sdf.withColumn("review_month", month(col("review_date")))
sdf = sdf.withColumn("review_yearmonth", sdf["review_date"].cast("timestamp").cast("string").substr(0, 7))

# Cast numerical columns
sdf = sdf.withColumn("helpful_upvotes", col("helpful_upvotes").cast("double"))
sdf = sdf.withColumn("helpful_total_votes", col("helpful_total_votes").cast("double"))

# Create binary label column
sdf = sdf.withColumn("label", when((col("rating") >= 5), 1.0).otherwise(0.0))

# ----------------------------------------
# Text Processing for review_summary
# ----------------------------------------
tokenizer = RegexTokenizer(inputCol="review_summary", outputCol="tokens", pattern="\\W")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
tf = HashingTF(inputCol="filtered", outputCol="text_tf", numFeatures=5000)
idf = IDF(inputCol="text_tf", outputCol="text_features")

# ----------------------------------------
# Assemble Final Features
# ----------------------------------------
numerical_features = ["helpful_upvotes", "helpful_total_votes", "review_detail_wordcount", "spoiler_tag"]
final_assembler = VectorAssembler(inputCols=["text_features"] + numerical_features, outputCol="features")

# ----------------------------------------
# Logistic Regression Model
# ----------------------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label")

# ----------------------------------------
# Pipeline Assembly
# ----------------------------------------
pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    tf,
    idf,
    final_assembler,
    lr
])

# ----------------------------------------
# Train-Test Split
# ----------------------------------------
train_data, test_data = sdf.randomSplit([0.8, 0.2], seed=42)

# ----------------------------------------
# Cross-Validation and Grid Search
# ----------------------------------------
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Set up CrossValidator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4  # Match with your number of cores
)

# Fit the cross-validated model
cvModel = crossval.fit(train_data)

# ----------------------------------------
# Predict & Evaluate
# ----------------------------------------
predictions = cvModel.transform(test_data)
predictions.select("review_summary", "label", "prediction").show(15, truncate=False)

roc_auc = evaluator.evaluate(predictions)
print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")

# --------------------------------------------
# Save Outputs to trusted/ and model/ folders
# --------------------------------------------

# Save feature-engineered training data
sdf.write.mode("overwrite").parquet("gs://my-bigdata-project-ky/trusted/feature_engineered_data.parquet")

# Save predicted data
predictions.write.mode("overwrite").parquet("gs://my-bigdata-project-ky/trusted/predictions.parquet")

# Save model
cvModel.write().overwrite().save("gs://my-bigdata-project-ky/models/LogisticRegression_model")





