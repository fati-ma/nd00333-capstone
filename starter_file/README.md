# Heart Failure Prediction

*TODO:* Write a short introduction to your project.

In this projects I used the *Heart Failure Prediction* dataset form Kaggle. I used Azure Machine Learning Service and Jupyter Notebook to train models using AutoML and HyperDrive and then by comparing the models performane I deployed the best among them as a HTTP REST endpoint and then tested it by sending a POST request.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

I got this dataset from Kaggle. In this dataset there are 12 features that can be used to predict mortality by heart failure, and the target is the **DEATH_EVENT** column that has two values; 1 means the patient died during the follow-up period and 0 means the person still alive/dropped out of the study. The dataset consists of 300 records/rows and 12 features/columns.
To know further and download/explore the dataset, use this [link](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

**Credit of dataset**
Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020)




### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

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
*TODO*: Explain how you are accessing the data in your workspace.

After downloading the dataset from kaggle as a csv file, I registered it as a dataset in the Azure Workspace in a Tabular form uploading from local system. 
XXXX I have used Dataset.get_by_name(ws, dataset_name) to access the registered dataset.
XXXXWe used method from_delimited_files of the TabularDatasetFactory Class to retreive data from the csv file 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

**AutoML** is the process of automating time consuming tasks of ML model development. It is used to build ML models with high efficiency. Unlike traditional ML model development that is rquires so much resources and time to produce and compare big number of models.

XXXX Configuration and settings used for the Automated ML experiment are described below:

| Configuration        | Value           | Description  |
| :-------------: |:-------------:| :-----:|
| compute_target      |  |  |
| task      |       |    |
| training_data |       |     |
| label_column_name |       |     |
| n_cross_validations |       |     |

| Settings        | Value           | Description  |
| :-------------: |:-------------:| :-----:|
| experiment_timeout_minutes      |  |  |
| enable_early_stopping      |       |    |
| iteration_timeout_minutes |       |     |
| max_concurrent_iterations |       |     |
| primary_metric |       |     |


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

The best performing model after training using AutoML is _______ with the Accuracy of ________________.

XXX parameters

XXX Improvements for autoML
- Using different metric other than *Accuracy*
- Increasing the value of*n_cross_validations* to reduce bias.


XXX screenshots

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

*Logistic Regrssion* algorithm was used. There are two hyperparamters for this experiment, C and max_iter. C is the inverse regularization strength whereas max_iter is the maximum iteration.

I have used *Random Parameter* sampling to sample over a discrete set of values. Hyperparameter values are randomly selected from the search space, where it chooses the values from a set of discrete values or a distribution over a continous range besides easy execution with minimal resources. For this problem, the hyperparameters that were given in search space are C (continuous) and max_iter(discrete). The hypermarameters:

 XXX  - A uniform distribution of values between 0.1 and 1 for Inverse of Regularization Strength: C
 XXX  - The Maximum Number of Iterations: max_iter between a range of 100 and 200

*BanditPolicy* was the one used. It terminates based on slack factor and evaluation interval which will terminate any run that doesn't fall within the specified slack factor .
Accuracy is evaluated using hyperDrive early stopping policy. The experiment will stop excution if conditions specified by the policy are met.

XXX In this experimentm the configurations used were evaluation_interval=1, slack_factor=0.2, and delay_evaluation=5. 


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

XXX The best accuracy that was achieved by the *Logistic Regression* model is _____

XXX Parameters : Regularization Strength (C)	2.0
Max Iterations (max_iter)	150

XXX Improvements for hyperDrive
- Using different metric other than *Accuracy*
- Using Bayesian Parameter Sampling instead of Random Sampling.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

XXX To deploy our Model using Azure ML we should first have a **trained model** then we will use **Inference Configuration** and pass it **entry_script** which is the scoring script that describes the input data and passes it to the model for prediction and then returns the results.
To download the scoring script; ```best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'scoreScript.py').```
We will also pass **environment** to *Inference Configuration*
To download the yml file associated with the environment: ```best_run.download_file('outputs/conda_env_v_1_0_0.yml', 'envFile.yml')```

Then for deployment configuration we will use Azure Container Instance (ACI) and pass it `cpu_cores = 1` and `memory_gb = 1`
1. A trained Model
2. Inference configuration; includes scoring script and environment
3. Deploy configuration; includes choice of deployment (ACI, AKS or local) and cpu/gpu/memory allocation

Then The test data passed to the model endpoint should be converted to JSON
![]()

Then the script will pass the test data to the model as a **POST** request and return the response using ```response = requests.post(service.scoring_uri, test_sample, headers=headers)```
![]()

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
