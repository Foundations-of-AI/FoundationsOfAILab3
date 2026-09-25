### In part 2 of this lab, you'll get your first exposure to scikit-learn, which is
## confusingly referred to as sklearn. sklearn is an excellent open-source machine learning
## package that helps with a lot of the pipeline setup needed with ML.

## To begin, let's load a dataset.

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay


# The iris dataset is a classic machine learning dataset.
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# This returns an object of type 'bunch'

iris = load_iris()
X, y = iris.data, iris.target

# X is the features, and y is the classification.

## To start, let's do the cross-validation mostly by hand.
scores = []
## the KFold object helps us here. It will automatically split the data.
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X) :
    ## Notice the multiple assignment - a popular Python trick.
    X_train, X_test, y_train, y_test = \
        (X[train_index], X[test_index], y[train_index], y[test_index])
    ## create a classifier.
    clf = tree.DecisionTreeClassifier(criterion="entropy")

    ## 'fit' is the terminology that sklearn uses for training.
    clf.fit(X_train, y_train)

    ## 'score' will compute the accuracy for each fold.
    ## It actually calls 'predict' for each element, which is the sklearn
    ## term for 'classify'.

    scores.append(clf.score(X_test, y_test))
print(scores)

## Question 1. Change the number of folds to 10 and 20. Does the accuracy change? How much?

## Question 2. Change the classifier to be a RandomForest with 50 estimators.
## RandomForestClassifier(n_estimators=50) How does the performance change with five-fold cross-validation?

### Hyperparameter search. The number of estimators is a hyperparameter.
## Almost every ML model has some sort of hyperparameters. Figuring out the
## best hyperparameters is a search problem.

## Let's create a dictionary containing two models. RandomForest and HistogramGradientBoosting.
## We don't need to worry too much about how these work for now. The basic idea is that with
## RandomForest, we create a set of independent decision trees, and with gradient boosting, we
## create a series of decision trees in which each builds off of the last.
## In this case we want to know which classifier is better, but we also have hyperparameters to deal
## with.

models = {
    "Random Forest": RandomForestClassifier(
        min_samples_leaf=5, random_state=0
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_leaf_nodes=15, random_state=0, early_stopping=False
    ),
}

## For RamdomForest, let's vary the number of estimators, and for boosting we'll vary
## the number of iterations the algorithm runs.

param_grids = {
    "Random Forest": {"n_estimators": [2,5,10]},
    "Hist Gradient Boosting": {"max_iter": [10, 20, 50]},
}

## create our KFold.
cv = KFold(n_splits=10, shuffle=True, random_state=0)

results = []
for name, model in models.items():
    ## The gridsearchcv object is going to help us out. It will automatically
    ## use the KFold object and the list of hyperparameter values and do grid search.
    ## In other words, it will do 10-fold cross-validation for each possible value and collect
    ## the results.

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        return_train_score=True,
        cv=cv,
    ).fit(X, y)
    result = {"model": name, "cv_results": pd.DataFrame(grid_search.cv_results_)}
    results.append(result)

print(results)

## Question 3. Adapt this to test 5, 10, 15, and 20 estimators for
# Random Forest and 25, 50, 75, and 100 iterations for Histogram Boosting,
# using five-fold cross-validation. What combination of model and hyperparamters
# gives the best accuracy?


### Pipelines
## Now let's get back to the SF Tree dataset, which is more interesting.

df = pd.read_csv('San_Francisco_Street_Tree_Inventory_20260921.csv', low_memory=False)
df.shape

## Recall from our last lab that the dataset is rather messy. In particular, there are
## some missing values.
## sklearn has some great tools to help us out here. We're going to use them to construct a
## Pipeline object, and then let that do the work for us.

## To begin, let's create two pipelines, one for numeric data and one for categorical data.

numeric_features = ['mapdbh']
categorical_features = ['planttype', 'planter', 'analysis_neighborhood']

## for numeric data, we'll use median to impute missing values.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

## For categorical variables, we'll do two things.
## first, we'll use the most frequent category for missing data.
## second, we're going to "one-hot encode" the categories.
## What this means is that instead of category names, we'll
## add a binary array, with a 1 in the position representing that category and
## a 0 everywhere else. This is much easier for many learning algorithms.
## e.g. if our categories were {'tall','short'} we would replace them with
## 10 or 01.

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
])

## Then let's hook these together with a ColumnTransformer.
## This transforms the specified columns according to the
## particular pipeline.

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

## set the output to be a pandas dataframe
preprocessor.set_output(transform="pandas")
### Question 1: 'species' has over 600 categories. This seems like a
### problem for one-hot encoding. What would you suggest as a strategy to
### make this more manageable?

### Question 2: Let's add the 'analysis_neighborhood' to the categorical features.
## Then, run fit_transform. This will first 'fit' the missing data (find the medians and most
## frequent, and then 'transform' the data by actually running it through the pipeline.

result = preprocessor.fit_transform(df)

## How many columns are in the resulting dataset? Does this make sense to you, given what
## you know about the 'analysis_neighborhood' and 'planter' categories?



### Now let's see how we can use sklearn to do an actual machine learning problem.
### Let's see what we can do about predicting the 'legalstatus' field. In particular,
### we want to predict the trees that are *not* DPW Maintained.
### Let's take a look at the data:

df['legalstatus'].value_counts()

## This dataset is very unbalanced! That means that we need a classifier that is good at
# identifying rare examples.
### let's make it a little easier and just deal with the six most common classes.

top_six = df['legalstatus'].value_counts().nlargest(6).index.tolist()
filtered_data = df[df['legalstatus'].isin(top_six)].copy()
filtered_data = filtered_data.dropna(subset=['legalstatus'])

### Question 3: This gives us a new dataset with only the six most common classifications.
## What percentage of the data did we just drop? We could have alternatively grouped them all
## into 'Other'. Which do you think is better?


### To begin, we need a baseline to compare our performance against.
### The dummy classifier (sometimes called Zero-R, for 'no rule') always picks the most
### common category. If we can't beat that, then there is no learnable pattern in the features.

### Our input features
numeric_features = ['mapdbh']
categorical_features = ['planttype', 'planter', 'analysis_neighborhood']
### the variable we want to predict
target = 'legalstatus'

X = filtered_data[numeric_features + categorical_features]
y = filtered_data[target]

## divide into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

## set up our pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]), categorical_features),
])

# Baseline: always predict the most frequent class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print('Baseline (dummy) accuracy:', dummy.score(X_test, y_test))

## note: no need to do cross-validation here.

### Question 4. The dummy gets 95% accuracy! Why is this a misleading number?


### Now let's fit an actual model. We're going to use RandomForest again.
### But we'll tell it to give all the classes equal weight regardless of frequency
### when computing error by setting class_weight='balanced'

## We can give the Pipeline class a list of steps in the constructor.
## This tells it to run the preprocessor (which has several Pipelines in it)
## then train the model, a RandomForestClassifier with fixed hyperparamters.
### (we could load this into GridSearchCV if needed)

tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, class_weight='balanced'
    )),
])

tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

print('Model accuracy:', accuracy_score(y_test, tree_pred))
print()
print(classification_report(y_test, tree_pred))

### Look at the recall for the minority classes. Let's compare that to the
## dummy classifier.

dummy_pred = dummy.predict(X_test)
print(classification_report(y_test, dummy_pred))

### We're doing better on the rare classes.
## Let's look at the confusion matrix.

ConfusionMatrixDisplay.from_predictions(tree_pred, y_test)
plt.show()

## Question 5. Apart from DPW maintained, which classes get confused most often?

### Question 6. Make a pipeline for a logisitic regression classifier,
### with LogisticRegression(class_weight='balanced', max_iter=1000), using the same preprocessor
## as above. Compare their performance. Which one would you recommend, and why?


log_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000, class_weight='balanced'
    )),
])

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print('Model accuracy:', accuracy_score(y_test, log_pred))
print()
print(classification_report(y_test, log_pred))

