#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import findspark
findspark.init()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix


# In[3]:


spark=SparkSession.builder .master ("local[*]").appName("part3").getOrCreate()


# In[4]:


sc=spark.sparkContext
sqlContext=SQLContext(sc)


# In[6]:


df=spark.read  .option("header","True") .option("inferSchema","True") .option("sep",";") .csv("Mini Project 2\data\XYZ_Bank_Deposit_Data_Classification.csv")


# In[7]:


print("There are",df.count(),"rows",len(df.columns),
      "columns" ,"in the data.") 


# In[8]:


#replace periods in column names with underscores
df = df.toDF(*(c.replace('.', '_') for c in df.columns))

df.toPandas().head(4)


# In[9]:


str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
print("Total catagorical Columns: " , len(str_cols))
print(str_cols)

dbl_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType) | isinstance(f.dataType, IntegerType)]
print("Total numerical Columns: ", len(dbl_cols))
print(dbl_cols)


# In[10]:


for col in df.columns:
    print(col, ":\n ", df.filter(df[col]=="?").count(), "null values\n ", df.select(col).distinct().count(), " distinct values")
    print


# In[11]:


#distribution of features
from matplotlib import cm
fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Distribution of Features", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(df.toPandas().describe().columns, range(1,11)):

    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=15)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()


# In[13]:


from pyspark.sql.functions import isnan, when, count, col
print(df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head())


# In[14]:


categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'y', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="vectorized_features", handleInvalid="skip") # use setHandleInvalid("skip")
stages += [assembler]
scaler = StandardScaler(inputCol="vectorized_features", outputCol="features")
stages += [scaler]

cols = df.columns
print(cols)
print(stages)


# In[15]:


pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

print(df.toPandas().head())


# In[16]:


# train/test split
train, test = df.randomSplit([0.8, 0.2], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

train.groupby("y").count().show()


# In[19]:


# Logistic Regression Model


# In[18]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import RandomForestRegressionModel, RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=5)
rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'label')

lrModel = lr.fit(train)
rfModel = rf.fit(train)

lrpredictions = lrModel.transform(test)
rfpredictions = rfModel.transform(test)
#predictions_train = lrModel.transform(train)
print("Logistic Regression")
print(lrpredictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5))
print("Random Forest")
print(rfpredictions.select("prediction", "label", "features").toPandas().head(5))

# End Model


# In[20]:


class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[21]:


y_true = lrpredictions.select("label")
y_true = y_true.toPandas()

y_pred = lrpredictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

accuracy = lrpredictions.filter(lrpredictions.label == lrpredictions.prediction).count() / float(lrpredictions.count())
print("Accuracy : ",accuracy)

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[26]:


class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
y_true = lrpredictions.select("label")
y_true = y_true.toPandas()

y_pred = lrpredictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()

accuracy = lrpredictions.filter(lrpredictions.label == lrpredictions.prediction).count() / float(lrpredictions.count())
print("Accuracy : ",accuracy)

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[28]:


#decision tree
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
import pandas as pd


dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dtc = dtc.fit(train)

pred = dtc.transform(test)
pred.show(3)

evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
acc = evaluator.evaluate(pred)
print("Prediction Accuracy: ", acc)

y_pred=pred.select("prediction").collect()
y_orig=pred.select("label").collect()

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)


# In[29]:


from pyspark.sql import SparkSession 
from pyspark.sql.functions import * 
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.classification import NaiveBayes 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[35]:


from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees


# In[41]:


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd 


# In[ ]:


# Gradient Boosted Tree


# In[42]:


gbtr = GBTRegressor(featuresCol='features', labelCol='label', maxIter=10)
gbtr = gbtr.fit(train)


# In[46]:


mdata = gbtr.transform(test)
mdata.show()

 

rmse=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse=rmse.evaluate(mdata) 

 

mae=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
mae=mae.evaluate(mdata) 

 

r2=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2=r2.evaluate(mdata)

evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
acc = evaluator.evaluate(pred)
print("Prediction Accuracy: ", acc)

y_pred=pred.select("prediction").collect()
y_orig=pred.select("label").collect()

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

print("RMSE: ", rmse)
print("MAE: ", mae)
print("R-squared: ", r2)


# In[47]:


#Gradient Boosted Tree viz

x_ax = range(0, mdata.count())
y_pred=mdata.select("prediction").collect()
y_orig=mdata.select("label").collect()

plt.plot(x_ax, y_orig, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Deposit test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# In[48]:


# Linear SVM Model


# In[53]:


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
import pandas as pd


# In[54]:


lsvc = LinearSVC(labelCol="label", maxIter=50)
lsvc = lsvc.fit(train)

pred = lsvc.transform(test)
pred.show(3)


# In[55]:


evaluator=MulticlassClassificationEvaluator(metricName="accuracy")
acc = evaluator.evaluate(pred)
 

print("Prediction Accuracy: ", acc)

y_pred=pred.select("prediction").collect()
y_orig=pred.select("label").collect()

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

