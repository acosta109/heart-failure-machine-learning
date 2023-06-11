# Predicting Heart Disease Using Machine Learning
## Team: Anthony Costa, Jonathan Diaz, Sarah Kim, Aakash Nagalapura
### [Data Set](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

# Overview of the Analysis
Our aim to produce a machine learning model that evaluates 11 factors to determine if someone has heart disease. 

* Age
* Sex
* ChestPainType
* Resting BP 
* Cholesterol
* FastingBS
* RestingECG
* MaxHR
* Exercise Angina
* Oldpeak
* ST_Slope
 
To get a general overview of our data we used `df.describe().T` and observed the following:

* `Avg. Age = 53.5`
* `Avg. RestingBP = 132`
* `Avg. Cholesterol = 199`
* `Avg. MaxHR = 25.5`
* `Avg. Oldpeak = 0.887`
* `Avg. HeartDisease = 0.553`

### Sex Ratio of the Data
![Sex Ratio](https://github.com/acosta109/heart-failure-machine-learning/assets/119609975/5ca7702c-32b4-4375-87f1-42cfe58f9939)
### Heart Disease Ratio by Sex
![heart disease ratio](https://github.com/acosta109/heart-failure-machine-learning/assets/119609975/853b48ca-2539-4d64-ab64-e89b61211409)
### Correlation Chart of the Data
![correlation](https://github.com/acosta109/heart-failure-machine-learning/assets/119609975/a1c8e2e8-4c1d-4cd8-bfa0-e5256af240da)

We first use decisions tress to create our model `clf=DecisionTreeClassifier(criterion="entropy")`. We found a maximum accuracy with `fold=3`.  Next, we decided to create a model for our dataset using RandomTree `clf=RandomForestClassifier(n_estimators=200,criterion="entropy")`. 
