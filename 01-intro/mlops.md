# MLOps Introduction

## What is MLOps?

Best practices for putting machine learning to production

### Process for ML Projects

Design

* If ML is right tool, use ML

Train

* Train the ML model

Operate

* Deploy the model for use

## Why is MLOps essential?

Automate performance of ML models

* If the performance of the model degrades, how do you retrain the model in a cost-effective manner?
* How do you monitor the model's performance in production and check that it is meeting expectations?

## MLOps Maturity Model

Source [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

|Level|Description|Overview|Use Case|
|-----|----|----|----|
|0| No Automation | No automation, all code in Jupyter notebooks| Proof of Concept |
|1| DevOps without MLOps| automated releases, unit and integration tests, continuous integration/continuous deployment (CI/CD), operational metrics, no experiment tracking, no reproducibility, data scientists work separately from engineers | Proof of Concept to Production |
|2| Automated Training | training pipeline, experiment tracking, model registry, low friction deployment, data scientists work with engineers| 2-3 or more ML cases |
|3| Automated Deployment | easy to deploy model, A/B testing, model monitoring| Many models in production |
|4| Full MLOps Automation | automated training and deployment, all of the above included | As needed |
