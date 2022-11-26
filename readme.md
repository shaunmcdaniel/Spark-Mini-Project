# Mini Project 2
## XYZ Bank Direct Marketing Campaign

## Team Data Cowboys
Destry Smith, Shaun Mcdaniel, Chris Brady, Ally Clifft

## Introduction
XYZ Bank has provided us with the company's direct marketing campaign data. The bank is interested forecasting whether or not a customer will sign up for a term deposit. They have asked us to find important relationships between variables in the data, as well as patterns and correlations between the target variable amd potential predictor variables. We will utilize Pyspark to conduct and Exploratory Data Analysis and build a collection of predictive models to provide XYZ Bank with valuable insights that will help them to predict the customers who are most likely to sign up for a term deposit. 

## Business Problem
Currently, XYZ Bank does not have an effective method to identify clients that will or will not subscribe for a term deposit. We are tasked with exploring the bank's telephone call data from past marketing campaigns, determining significant relationships between the input variables and the target variable, and uncover patterns within the data. This analysis aims to provide XYZ bank with actionable insights that will allow them to design future marketing campaigns that are more effective in targeting clients that are likely to subscribe for a term deposit. 

## Data
The data consists of information regarding the bank's past direct marketing campaigns. The data consists of 20 variable columns that are sorted by date between May 2008 and November 2010. The varibales provide information regarding clients' demographics, most recent contact with the bank concerning the campaign, broad social and economic statistics, and other information regarding the overall marketing campaign efforts. 

Specific data attributes:
![image](https://user-images.githubusercontent.com/111790692/202341830-f70df4bd-7376-4451-98c6-e6f09bd8067f.png)

## Exploratory Data Analysis
The first step in our approach to understand the campaign data was to complete exploratory testing. We began by importing a series of necessary Pyspark packages into the Python environment. 

Once all required packages were imported, we loaded the data into a data frame. The data frame will allow for a smoother analysis process moving forward. Once the data frame was created, we gathered the total count of rows and columns, which revealed that there are 41188 rows and 21 columns of data.
![image](https://user-images.githubusercontent.com/111790692/202345174-009f53e3-6b42-4bfd-8294-554792fe3672.png)

Next, we began to clean and preprocess the data. We began by replacing all periods in the column names with underscores. After cleaning the column headers, we identified the totla number of categorical and numerical columns in the data set. Each variable column was tested for cardinality, as well as null values. Each categorical variable was ran through an early analysis to view the distribution of its features. In order to ensure that the data contained no null values, we double checked the null values after the feature distribution analysis and prior to the test and train data split. 

Replaced column names: ![image](https://user-images.githubusercontent.com/111790692/202346531-1661a788-967b-4bfa-ad16-b1102eb68639.png)

Identify column types: ![image](https://user-images.githubusercontent.com/111790692/202346642-08eece30-aa8c-408e-b89e-92d230e2f574.png)

Cardinality & null values: 
![image](https://user-images.githubusercontent.com/111790692/202346714-9c1a7777-2081-4bc9-9e41-f53a61dd22eb.png)
![image](https://user-images.githubusercontent.com/111790692/202346765-0286be3a-5a8d-4781-ac79-55fbe74a08a5.png)

Distribution of features: ![image](https://user-images.githubusercontent.com/111790692/202346928-785131ac-0885-4d2b-963b-eedfe0676781.png)

Double check for null values:![image](https://user-images.githubusercontent.com/111790692/202346984-9d49418b-0440-4cf4-82c8-a4632027463b.png)

Test/train split: 
![image](https://user-images.githubusercontent.com/111790692/202347026-da966968-2d3a-40d8-b8a3-4877de0bd189.png)

After the initial cleaning of the data, we began preprocessing through One Hot encoding the categorical features, vectorizing & scaling all features, and adding the models we tested. 
![image](https://user-images.githubusercontent.com/111790692/202347886-15fd3dc4-cf2f-4b62-b092-e78727c4d56b.png)

One the initial cleaning and processing of the data was complete. Each member of our team worked on testing and designing models for the data. 

## Modelling
Prior to model prediction, we trained each model and found the prediction and probaility statistics for each. They can be seen below. 

![image](https://user-images.githubusercontent.com/111790692/202356174-cbb61482-5d3c-48e0-a8b7-5d25bfcadce0.png)
![image](https://user-images.githubusercontent.com/111790692/202356204-595e8074-f8a8-48eb-9b31-2ff0c05b861b.png)
![image](https://user-images.githubusercontent.com/111790692/202356225-c82a8656-9984-41ad-953b-b2b914b67f5b.png)


In order to achieve the best-fit model and provide XYZ Bank with the most accurate predictive analysis, we designed and tested the data on 6 different models. We developed confusion matrices and ROC curves for each individual model, with the exception of the K-Means clustering model.  
- Linear Regression Model: This model had a 91% accuracy in identifying relationships and patterns between variables. The model possessed an AUC of 0.935, which is an excellent measure of the model's accuracy in performing across different thresholds. 
![image](https://user-images.githubusercontent.com/111790692/202349306-1819b6d1-4692-4214-8a2a-d7dada421f58.png)
![image](https://user-images.githubusercontent.com/111790692/202349352-e5c21cc7-7fbb-4d71-9b09-a38cb0d83ab9.png)

- Random Forest Model: This model possesses an 90% accuracy and a 0.915 AUC measure. Although both of these numbers depict that the model is accurate and performs well across thresholds, it is not quite as strong as the linear regression model. 
![image](https://user-images.githubusercontent.com/111790692/202349833-749bd6c0-08fc-4261-833c-1c4f3c0fe49c.png)
![image](https://user-images.githubusercontent.com/111790692/202349882-04f0c901-d5dc-4362-97e2-cef8e3bab7e4.png)

- Decision Tree Model: The decision tree model held a higher accuracy rating than the linear regression model by 1%. This model's accuracy was 92%. The model received an AUC rating of 0.931, which, like the linear regression model, suggests that it has a strong performance against different thresholds. 
![image](https://user-images.githubusercontent.com/111790692/202350174-cf5ce213-6dbb-4065-b313-6ed39b5fc472.png)
![image](https://user-images.githubusercontent.com/111790692/202350212-c645ab6d-6a60-4ab4-8fb2-4b50ae75d7da.png)

- Gradient Boosted Tree Model: The gradient boosted tree stood at an accuracy of 92% and held an AUC score of 0.932. 
![image](https://user-images.githubusercontent.com/111790692/202350824-a17345b0-d11d-4ae2-a9a3-ec0a880ddaec.png)
![image](https://user-images.githubusercontent.com/111790692/202350853-7d8d3c5b-25bb-4523-b067-29b1a8b57fe7.png)

- Linear SVC Model: This model held the same accuracy as the Random Forest model at 90%. However, it did possess a stronger AUC at a score of 0.932.
![image](https://user-images.githubusercontent.com/111790692/202350590-e77bcdf9-e4cd-4fde-9a45-8d0ff90f3d8b.png)
![image](https://user-images.githubusercontent.com/111790692/202350613-8f78fc57-b084-438c-b30e-3e47c56cda73.png)

Overall, each of the above 5 models possessed extremely high accuracy ratings, all falling above the 90% mark. 
![image](https://user-images.githubusercontent.com/111790692/202351187-f9275d70-5cce-431b-b86e-c65724ff241e.png)

Based on the confusion matrices and ROC curves, we moved to evaluate the Gradient Boosted Tree model, as it held the best accuracy statistics. In order to evaluate the model, we set up hyper-parameters and created a 'ParamGrid' for cross validation purposes. The following statistics were found for the Gradient Boosted Tree: 

Best Model Test AUC: 
![image](https://user-images.githubusercontent.com/111790692/202351685-068a4dc2-573d-4d06-ac9f-0049c77ee201.png)


Feature Weights:
![image](https://user-images.githubusercontent.com/111790692/202351756-fe7f0050-a84f-458f-86f1-619d5e177d77.png) ![image](https://user-images.githubusercontent.com/111790692/202351798-367eef5e-903c-448d-b03c-0a537f8c68f1.png) ![image](https://user-images.githubusercontent.com/111790692/202351820-4cc64961-a029-4fd0-b7bc-2924cb808be4.png)

Upon receiving the best model statistics, we extracted all important features in orderl to build a prediction chart. A sample of the new prediction columns can be seen below. 
![image](https://user-images.githubusercontent.com/111790692/202352928-2901af08-9c75-4228-a804-b47bb0dadcad.png)


- K-Means Clustering: In completing the K-Means clustering, we followed the same cleaning and preprocessing steps that are detailed in the section above. 
KMeans is an unsupervised model that attempts to group similar data points together in a cluster. New data points can be related to one of the clusters. Marketing materials based on the attributes of the cluster would be sent to customers, making for an effective marketing campaign. For this project, we first attempted to find the appropriate number of clusters to use in the model. This was done determining the silhouette score of clusters from counts from 2 to 25. The silhouette score shows how close the data points of a cluster are to their center and how separated they are form other clusters. The silhouette score ranges from -1 to 1. The higher the score the better. Out of 22 clusters there two that received a score of 0.215, the rest were all under 0.2. The max score was at 22 clusters. Thinking that 22 clusters may be too many, we decided to look for the highest score between 3 and 10. The cluster count that was selected to be tested was 9. Seeing the silhouette scores we got made us think that our k-means model may not be the best model for this project. Next we created a graph using matplotlib.pyplot to look at cluster count vs silhouette score.

The third step was to generate the model using kmeans from pyspark.ml.clusterting. We used our cluster count of 9 and a seed of 1. Once the model was generated, it was run against the test data. The model created a column named prediction that showed which cluster the record belonged to. Finally, a group by was run against the prediction column to get a count of how many records belonged to each cluster.

## Recommendations
The goal was to identify clients who would be most likely to sign up for term deposits based on their previous marketing data it could be beneficial to find the clients they contact most frequently. Also, since we know they have contact information for these clients, the company could then find a portion of their client base who are contacted the most; the top 25% of most contacted as an example. They could then start sending these clients more information to these clients about long term deposit products to see how well the products perform over a set period of time. The company could utilize an automated email system with a filter for the clients they want to include to be more efficient. Depending on the outcome of this, the company could then start pushing out similar emails and correspondces to their entire client base and test the performance of those products
