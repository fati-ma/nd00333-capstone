# Heart Failure Prediction

In this projects I used the *Heart Failure Prediction* dataset form Kaggle. I used Azure Machine Learning Service and Jupyter Notebook to train models using AutoML and HyperDrive and then by comparing the models performane I deployed the best among them as a HTTP REST endpoint and then tested it by sending a POST request.

## Dataset

### Overview

I got this dataset from Kaggle. In this dataset there are 12 features that can be used to predict mortality by heart failure, and the target is the **DEATH_EVENT** column that has two values; 1 means the patient died during the follow-up period and 0 means the person still alive/dropped out of the study. The dataset consists of 300 records/rows and 12 features/columns.
To know further and download/explore the dataset, use this [link](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

**Citationt**
Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020)




### Task

This problem is a ***Classification*** problem, we have to predict either one of the two states (Died during the follow up period/ Still alive), the features I will use to predict mortality by heart failure are explained below:

The **features** are:
- Age: Age of the patient in years between 40 and 95 years
- Anaemia: Decrease of red blood cells or hemoglobin - Boolean  
- High blood pressure: If a patient has hypertension - Boolean
- Creatinine phosphokinase (CPK): Level of the CPK enzyme in the blood  
- Diabetes: If the patient has diabetes - Boolean
- Ejection fraction: Percentage of blood leaving between 14 and 80 percent
- Sex: Woman or man - Binary - 0: Female  1: Male
- Platelets: Platelets in the blood          
- Serum creatinine: Level of creatinine in the blood  
- Serum sodium: Level of sodium in the blood    
- Smoking: If the patient smokes - Boolean  
- Time: Follow-up period in days between 4 and 285 days

### Access

After downloading the dataset from kaggle as a csv file, I registered it as a dataset in the Azure Workspace in a Tabular form uploading from local file. 
I used method `from_delimited_files` of the `TabularDatasetFactory` Class to retreive data from the csv file in `train.py` and the path [https://raw.githubusercontent.com/fati-ma/nd00333-capstone/master/heart_failure_clinical_records_dataset%5B1%5D.csv](https://raw.githubusercontent.com/fati-ma/nd00333-capstone/master/heart_failure_clinical_records_dataset%5B1%5D.csv)

## Automated ML

**AutoML** is the process of automating time consuming tasks of ML model development. It is used to build ML models with high efficiency. Unlike traditional ML model development that is rquires so much resources and time to produce and compare big number of models.

Configuration and Settings used for the Automated ML experiment are described below:

```

automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 4,
    "n_cross_validations": 3,
    "primary_metric" : 'accuracy'
}


automl_config = AutoMLConfig(
                             compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT", 
                             enable_early_stopping= True,
                             **automl_settings
```                             

| Settings        | Value           | Description  |
| :-------------: |:-------------:| :-----:|
| experiment_timeout_minutes      | 20 | The amount of time the experiment takes before it terminates |
| max_concurrent_iterations |   4    |  The max number of iteration that can be executed in parallel   |
| n_cross_validations      |    3   |  To avoid falling in overfitting  |
| primary_metric |    'accuracy'   |  The metric that will be optimized for model selection   |


| Configuration        | Value           | Description  |
| :-------------: |:-------------:| :-----:|
| compute_target      | compute_target | used to run the experiment on |
| task      |    "classification"    |  because the problem has binary predictions |
| training_data |   dataset    |   contains both features and label columns  |
| label_column_name |   "DEATH_EVENT"    |  The name of the target column   |
| enable_early_stopping |   True    |     |


**The models that were trained in AutoML**

![autoMl-models]()


**Run Details Widget** 

![run-details]()


### Results

The best performing model after training using AutoML is `VotingEnsemble` with the Accuracy of **0.8594276094276094**.

**Best model Metrics and Parameters:**

![best-model]()

![best-metrics]()

![best-params]()


**Improvements for autoML**
- Using different metric other than *Accuracy*
- Increasing the value of *n_cross_validations* to reduce bias.


**Screenshots of AutoML experiment**

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![register-model]()


## Hyperparameter Tuning


*Logistic Regrssion* algorithm was used. There are two hyperparamters for this experiment, C and max_iter. C is the inverse regularization strength whereas max_iter is the maximum iteration.

I have used *Random Parameter* sampling to sample over a discrete set of values. Hyperparameter values are randomly selected from the search space, where it chooses the values from a set of discrete values or a distribution over a continous range besides easy execution with minimal resources. For this problem, the hyperparameters that were given in search space are C (continuous) and max_iter(discrete). The hypermarameters:

  - A uniform distribution of values between 0.1 and 1 for Inverse of Regularization Strength: C
  - The Maximum Number of Iterations: max_iter between a range of 50 and 200

*BanditPolicy* was the one used. It terminates based on slack factor and evaluation interval which will terminate any run that doesn't fall within the specified slack factor .
Accuracy is evaluated using hyperDrive early stopping policy. The experiment will stop excution if conditions specified by the policy are met.

In this experimentm the configurations used were `evaluation_interval=2`, `slack_factor=0.1`, and `delay_evaluation=3`. 

HyperDriveConfig was created by specifying the estimator, hyperparameter_sampling and terrmination policy, primary_metric_name, max_total_runs besides other as shown in the image below:

![config]()



### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The best accuracy that was achieved by the `Logistic Regression` model is **0.7777777777777778**.

**Parameters** : 
- Regularization Strength (C)	: 0.6878054492330412
- Max Iterations (max_iter)	: 200


**Improvements for hyperDrive**
- Using different metric other than *Accuracy*
- Using Bayesian Parameter Sampling instead of Random Sampling.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

**Screenshots**

![run-details]()

![register-best-model]()

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Since the AutoMl model: `VotingEnsemble` got the highest accuracy, it was the one I chose to deploy.

To deploy our Model using Azure ML we should first have a **trained model** then we will use **Inference Configuration** and pass it **entry_script** which is the scoring script that describes the input data and passes it to the model for prediction and then returns the results.

To download the scoring script ```best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'score.py').```

We will also pass **environment** to *Inference Configuration*
To download the yml file associated with the environment: ```best_run.download_file('outputs/conda_env_v_1_0_0.yml', 'environment.yml')```

The AutoMl model is deployed using Azure Container Instance as a WebService, so for deployment configuration I used ACI and pass it `cpu_cores = 1` and `memory_gb = 1`

To consume the model the test data that was selected from the dataframe is passed and should be converted to JSON
![json-data]()

The model is successfully deployed as a webservice, a REST endpoint is created, the status Healthy and we have the scoring uri to test the endpoint and the swagger uri.

![deploy]()

## Screen Recording
The screencast demonstrates:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

This is the [screen recording](https://drive.google.com/file/d/1l-SsdZaxuax_kXFa4OrNeR-vpDNwJPAk/view?usp=sharing) of the project.

