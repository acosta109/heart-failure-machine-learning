# Predicting Heart Disease Using Machine Learning
## Team: Anthony Costa, Jonathan Diaz, Sarah Kim, Aakash Nagalapura
### [Data Set](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
### Jupyter Notebook
### PowerPoint Presentation


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
![correlation](https://github.com/acosta109/heart-failure-machine-learning/assets/119609975/21acb103-6c53-407b-8fa4-8904312b5230)



We first use decisions tress to create our model `clf=DecisionTreeClassifier(criterion="entropy")`. We found a maximum accuracy with `fold=2`.  Next, we decided to create a model for our dataset using RandomTree `clf=RandomForestClassifier(n_estimators=200,criterion="entropy")`.  Lastly, we created a LogisiticRegression model `clf=LogisticRegression()`.

## Results

* Machine Learning Model 1: 
  * `clf=DecisionTreeClassifier(criterion="entropy")`
  * `fold = 2`
  * `accuracy = 0.898`
  * `precision = No Heart Disease: 0.77 --- Heart Disease: 0.92`
  * `recall = No Heart Disease: 0.91 --- Heart Disease: 0.77` 
  * `f1-score = No Heart Disease: 0.83 --- Heart Disease: 0.84 `

* Machine Learning Model 2: 
  * `clf=RandomForestClassifier(n_estimators=200,criterion="entropy")`
  * `fold = 1`
  * `accuracy = 0.767`
  * `precision = No Heart Disease: 0.85 --- Heart Disease: 0.75`
  * `recall = No Heart Disease: 0.62 --- Heart Disease: 0.91` 
  * `f1-score = No Heart Disease: 0.72 --- Heart Disease: 0.82 `

![feature_imp](https://github.com/acosta109/heart-failure-machine-learning/assets/119609975/4fc03055-ab4d-4526-baae-88f3470d5dbd)


* Machine Learning Model 3: 
  * `clf=LogisticRegression()`
  * `fold = 1`
  * `accuracy = 0.889`
  * `precision = No Heart Disease: 0.79 --- Heart Disease: 0.91`
  * `recall = No Heart Disease: 0.90 --- Heart Disease: 0.80` 
  * `f1-score = No Heart Disease: 0.84 --- Heart Disease: 0.85 `

## Conclusion
