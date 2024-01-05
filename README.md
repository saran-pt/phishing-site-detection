# Phishing Website Detection by Machine Learning Techniques

## Objective

Phishing websites are a common social engineering tactic that imitates trustworthy URLs and webpages. This project aims to employ machine learning models and deep neural networks to predict phishing websites. A dataset containing both phishing and non-phishing URLs is utilized, from which features are extracted. The performance of each model is assessed and compared.

## Data Collection

Phishing URLs are collected from the PhishTank open-source service, which provides regularly updated sets of phishing URLs.

Legitimate URLs are obtained from the University of New Brunswick's open datasets: [UNB URL Dataset](https://www.unb.ca/cic/datasets/url-2016.html). The benign URLs from this dataset are used for this project.

## Models & Training

The dataset is split into 80-20 for training and testing. This project focuses on a supervised classification problem, categorizing URLs as phishing (1) or legitimate (0). The following machine learning models are employed:

- LogisticRegression
- MultinomialNB

All models are trained and evaluated against the test dataset.

