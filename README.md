
## Cardiotocographic Fetal Health Classification w/ ANN & SVM
***

Neural Computing [INM427]

#### Overview
This research explores the classification of fetal health conditions using Artificial Neural Networks (ANN) and Support Vector Machines (SVM) applied to cardiotocography (CTG) data in severely imbalanced clinical cases.

The CTG dataset from [UC Irvine's ML repository](https://archive.ics.uci.edu) contains 2126 instances of cardiotocograph records with 21 clinical features. Over three target classes (normal, susceptible, and pathological), data imbalance favoring the majority class accounts for approximately 77% of the overall instances.

To address the data imbalance issue, we used stratified sampling techniques in data partitioning as well as the [AdaSyn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html) Algorithm to oversample the susceptible and patholigical classes.
Furthermore, we also included pre-model feature selection using recursive feature elimination based on variance inflation factor (VIF) to reduce multicollinearity between predictors.

The training methodology also included K-Fold cross validation and Multi-step hyperparameter tuning using grid-search for both architectural hyperparameters as well as optimization hyperparameters.


While the SVM model achieved comparible results to the ANN model in overall macro-averaged and weighted-averaged metrics, the specific clinical context values more heavily on class-wise sensitivity given the data imbalance of the suspected and pathological classes. The final ANN model achieved an overall geometric mean of 0.92, specificity of 0.95, and F-measure of 0.89.

<br>

***


#### Directory Structure


```zsh
> CTG
    ├── data
    │   ├── CTG.xls
    │   └── test.csv
    ├── main.py
    ├── models
    │   ├── ann_params.pkl
    │   ├── ann_weights.pt
    │   └── svm.pkl
    ├── README.md
    ├── paper.pdf
    ├── requirements.txt
    └── test.py
```

<br>

#### Frameworks
main frameworks and packages used:

- [pytorch](https://pytorch.org) 
- [skorch](https://github.com/skorch-dev/skorch)
- [imbalanced-learn](https://imbalanced-learn.org/stable/)

<br>

#### Build

This project was developed using python version `3.12.2`\

To run the model on the test dataset `test.csv`:

    1. clone this repo
    2. cd into the local repo
    3. set up a virtual env and install dependencies via `requirements.txt`
    4. activate the environment and run script


In the command line:

```zsh
git clone <repo-url>
cd CTG          

uv venv --python 3.12.2
source .venv/bin/activate

uv pip install -r requirements.txt
uv run test.py
```

<br>

***

#### Acknowledgements

This research was conducted with the support of City, University of London as part of the INM427 Neural Computing module under the Dept. of Computer Science. 