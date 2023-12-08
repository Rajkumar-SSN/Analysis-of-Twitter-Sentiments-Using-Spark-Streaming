# Real-time Sentiment Analysis with Apache Spark Streaming

## Objective

The primary goal of this project is to implement a real-time sentiment analysis system using Apache Spark Streaming for Twitter data. Through this project, users can gain practical experience in setting up Spark Streaming applications, collecting and processing Twitter data, and analyzing sentiments in real-time.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Data Collection](#setup-and-data-collection)
   - [Environment Setup](#environment-setup)
   - [Data Collection](#data-collection)
3. [Data Processing and Preparation](#data-processing-and-preparation)
   - [Selecting Relevant Data](#selecting-relevant-data)
   - [Data Splitting](#data-splitting)
4. [Sentiment Analysis](#sentiment-analysis)
   - [Data Tokenization and Stop Words Removal](#data-tokenization-and-stop-words-removal)
   - [Converting Words to Numerical Features](#converting-words-to-numerical-features)
5. [Model Training](#model-training)
6. [Testing and Evaluation](#testing-and-evaluation)
   - [Data Preparation for Testing](#data-preparation-for-testing)
   - [Prediction and Accuracy Calculation](#prediction-and-accuracy-calculation)
7. [Applications of Sentiment Analysis](#applications-of-sentiment-analysis)
8. [Results and Conclusion](#results-and-conclusion)
9. [Future Enhancements](#future-enhancements)

## Introduction

Social media platforms like Twitter generate vast amounts of data, making them invaluable sources for understanding public sentiment. This project details the implementation of a real-time sentiment analysis system using Apache Spark Streaming, with the goal of processing Twitter data, classifying sentiments, and evaluating the model's accuracy.

## Setup and Data Collection

### Environment Setup

To run this project, a Spark session was created using the PySpark library, facilitating large-scale data processing in a distributed computing environment.

```python
# Sample code for Spark session setup
!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover

# create Spark session
appName = "Sentiment Analysis in Spark"
spark = SparkSession \
    	.builder \
    	.appName(appName) \
    	.config("spark.some.config.option", "some-value") \
    	.getOrCreate()
```

### Data Collection

Twitter data was collected using the Twitter Streaming API and stored in a CSV file, which was later ingested into a Spark DataFrame.

```python
# Sample code for Twitter data collection
from google.colab import drive
drive.mount("/content/drive")

#read csv file into dataFrame with automatically inferred schema
tweets_csv = spark.read.csv('/content/drive/MyDrive/Project/BigData_TeamProject/tweets.csv',              inferSchema=True, header=True)
tweets_csv.show(truncate=False, n=3)

```

## Data Processing and Preparation

### Selecting Relevant Data

The relevant columns, "SentimentText" and "Sentiment," were selected, and the "Sentiment" column was cast to integers.

```python
# Sample code for selecting relevant data
#select only "SentimentText" and "Sentiment" column,and cast "Sentiment" column data into integer
data = tweets_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
data.show(truncate = False,n=5)

```

### Data Splitting

The dataset was divided into training (70%) and testing (30%) sets.

```python
# Sample code for data splitting
#divide data, 70% for training, 30% for testing
dividedData = data.randomSplit([0.7, 0.3])
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing

```

## Sentiment Analysis

### Data Tokenization and Stop Words Removal

The "SentimentText" was tokenized into individual words, and stop words were removed.

```python
# Sample code for data tokenization and stop words removal
#Prepare training and testing data
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)

#Separate "SentimentText" into individual words using tokenizer
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(trainingData)
tokenizedTrain.show(truncate=False, n=5)

#Removing stop words (unimportant words to be features)
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(),outputCol="MeaningfulWords")
SwRemovedTrain = swr.transform(tokenizedTrain)
SwRemovedTrain.show(truncate=False, n=5)

```

### Converting Words to Numerical Features

Words were converted into numerical features using the HashingTF function.

```python
# Sample code for converting words to numerical features
hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
numericTrainData = hashTF.transform(SwRemovedTrain).select('label', 'MeaningfulWords', 'features')
numericTrainData.show(truncate=False, n=3)

```

## Model Training

A logistic regression model was trained using the prepared training data.

```python
# Sample code for model training
lr = LogisticRegression(labelCol="label", featuresCol="features",maxIter=10, regParam=0.01)
model = lr.fit(numericTrainData)
print ("Training is done!")

```

## Testing and Evaluation

### Data Preparation for Testing

The same preprocessing steps were applied to the testing data.

```python
# Sample code for data preparation for testing
# Preparing testing data
tokenizedTest = tokenizer.transform(testingData)
SwRemovedTest = swr.transform(tokenizedTest)
numericTest = hashTF.transform(SwRemovedTest).select('Label', 'MeaningfulWords', 'features')
numericTest.show(truncate=False, n=2)

```

### Prediction and Accuracy Calculation

The model was used to predict sentiments on the testing data, and accuracy was calculated.

```python
# Sample code for prediction and accuracy calculation
prediction = model.transform(numericTest)
predictionFinal = prediction.select("MeaningfulWords", "prediction", "Label")
predictionFinal.show(n=4, truncate = False)
correctPrediction = predictionFinal.filter(predictionFinal['prediction'] == predictionFinal['Label']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData,", accuracy:", correctPrediction/totalData)
```

## Applications of Sentiment Analysis

- Predict the success of a movie
- Predict political campaign success
- Decide whether to invest in a certain company
- Targeted advertising
- Review products and services

## Results and Conclusion

The sentiment analysis model demonstrated a certain level of accuracy on the testing data. The detailed process of data preprocessing, model training, and evaluation has been presented. Further enhancements and explorations, such as parameter tuning and additional feature engineering, can be considered for improving model performance.

## Future Enhancements

Proposing potential enhancements and future directions for the sentiment analysis system, such as incorporating advanced machine learning techniques or expanding the analysis to other social media platforms.

---

