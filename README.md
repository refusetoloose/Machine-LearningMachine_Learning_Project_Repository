# Machine Learning Project Repository

This repository serves as a practical resource for conducting machine learning activities focused on classification, regression, and clustering tasks. It provides datasets and skeleton code files for three activities:

Classifying Titanic Survival: Participants explore a dataset containing information about Titanic passengers and their survival status. They are tasked with building a predictive model to classify whether a passenger would have survived based on their characteristics.

Predicting House Prices: Participants work with the Ames Housing dataset, which includes various features of houses and their sale prices. They are challenged to develop a regression model to forecast the sale price of a house based on its characteristics.

Clustering Seeds: Participants use a wheat seed dataset with information about seed characteristics to build a clustering model. The goal is to group seeds into clusters based on their internal kernel structure.

The repository includes setup instructions, code skeletons, and guidelines for exploring the datasets, visualizing relationships between variables, and improving upon baseline models provided in the skeleton code.


## Introduction

In this activity, participants will practice machine learning techniques to solve real-world problems using three different datasets. They will perform classification, regression, and clustering tasks, respectively.

## Equipment and Materials

To run the code in this repository, you'll need the following:

- BYOD laptop
- Python
- Visual Studio Code
- TXT file: `requirements.txt`
- Git file: `.gitignore`
- Activity 1 data and skeleton files:
  - `titanic.zip`
  - `truth_titanic.csv`
  - `classification.py`
- Activity 2 data and skeleton files:
  - `house-prices-advanced-regression-techniques.zip`
  - `truth_house_prices.csv`
  - `regression.py`
- Activity 3 data and skeleton files:
  - `Seeds_dataset.txt`
  - `clustering.py`

## Setup

1. Download all the data and skeleton files listed above and create a folder for them on your computer.
2. Unzip each .zip file into the respective folder.
3. Open the folder in Visual Studio Code.
4. Create a virtual environment named `venv`, configure VS Code to use it as the default Python interpreter, and install the required dependencies listed in `requirements.txt`.

## Activities

### 1. Classifying Titanic Survival

#### Files:
- `titanic.zip`: Dataset containing information about Titanic passengers.
- `truth_titanic.csv`: Ground truth labels for Titanic survival.
- `classification.py`: Python code for building a predictive model to classify Titanic survival.

#### Description:
In this activity, we explore the Titanic dataset to predict whether a passenger survived the disaster. The `classification.py` file contains code for data exploration, model building, and evaluation. Participants are encouraged to improve upon the baseline model provided in the skeleton code.

### 2. Predicting House Prices

#### Files:
- `house-prices-advanced-regression-techniques.zip`: Dataset containing characteristics of houses and their sale prices.
- `truth_house_prices.csv`: Ground truth sale prices for houses.
- `regression.py`: Python code for building a regression model to predict house prices.

#### Description:
This activity involves predicting the sale prices of houses based on their features. The `regression.py` file contains code for data exploration, visualization, model development, and evaluation. Participants are challenged to build a regression model that outperforms the provided baseline.

### 3. Clustering Seeds

#### Files:
- `Seeds_dataset.txt`: Dataset containing information about wheat seeds.
- `clustering.py`: Python code for clustering wheat seeds based on their characteristics.

#### Description:
In this activity, participants use the wheat seed dataset to group seeds into clusters based on their internal kernel structure. The `clustering.py` file contains code for data exploration, visualization, determining the optimal number of clusters, and implementing KMeans clustering.

## Usage

1. Clone this repository to your local machine.
2. Navigate to the respective activity folder.
3. Open the Python files in your preferred editor (e.g., Visual Studio Code).
4. Follow the instructions within the code files to explore the datasets, build models, and evaluate performance.
5. Experiment with different approaches and algorithms to improve model performance.

## Dependencies

- Python
- pandas
- seaborn
- scikit-learn

## References

- Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P. A., Lukasik, S., & Zak, S. (2010). A complete gradient clustering algorithm for features analysis of X-ray images. In Pietka, E. and Kawa, J. (Eds.), Information technologies in biomedicine, (pp. 15–24). Berlin: Springer-Verlag.
- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. Journal of Statistics Education, 19(3). doi: 10.1080/10691898.2011.11889627

