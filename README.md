# Finding the Best Value for K
  
## Introduction
In this lesson, you'll investigate how changing the value for K can affect the performance of the model, and how to use this to find the best value for K.
## Objectives
•	Conduct a parameter search to find the optimal value for K
•	Explain how KNN is related to the curse of dimensionality
## Finding the optimal number of neighbors
By now, you've got a strong understanding of how the K-Nearest Neighbors algorithm works, but you likely have at least one lingering question—what is the best value to use for K? There's no set number that works best. If there was, it wouldn't be called K-nearest neighbors. While the best value for K is not immediately obvious for any problem, there are some strategies that you can use to select a good or near optimal value.
## K, overfitting, and underfitting
In general, the smaller K is, the tighter the "fit" of the model. Remember that with supervised learning, you want to fit a model to the data as closely as possible without overfitting to patterns in the training set that don't generalize. This can happen if your model pays too much attention to every little detail and makes a very complex decision boundary. Conversely, if your model is overly simplistic, then you may have underfit the model, limiting its potential. A visual explanation helps demonstrate this concept in practice:
![image](https://github.com/user-attachments/assets/99c77802-26ed-45d4-b2ba-ad2696f0b698)

When K is small, any given prediction only takes into account a very small number of points around it to make the prediction. If K is too small, this can end up with a decision boundary that looks like the overfit picture on the right.
Conversely, as K grows larger, it takes into account more and more points, that are farther and farther away from the point in question, increasing the overall size of the region taken into account. If K grows too large, then the model begins to underfit the data.
It's important to try to find the best value for K by iterating over a multiple values and comparing performance at each step.
 ![image](https://github.com/user-attachments/assets/b0f18135-ff80-4e73-8b20-2e9b9eb31f6e)

As you can see from the image above, k=1 and k=3 will provide different results!
## Iterating over values of K
Since the model arrives at a prediction by voting, it makes sense that you should only use odd values for k, to avoid ties and subsequent arbitrary guesswork. By adding this constraint (an odd value for k) the model will never be able to evenly split between two classes. From here, finding an optimal value of K requires some iterative investigation.
The best way to find an optimal value for K is to choose a minimum and maximum boundary and try them all! In practice, this means:
1.	Fit a KNN classifier for each value of K
2.	Generate predictions with that model
3.	Calculate and evaluate a performance metric using the predictions the model made
4.	Compare the results for every model and find the one with the lowest overall error, or highest overall score!
 ![image](https://github.com/user-attachments/assets/ad7ccae0-3971-4832-a843-8aaa76b500fc)

A common way to find the best value for K at a glance is to plot the error for each value of K. Find the value for K where the error is lowest. If this graph continued into higher values of K, we would likely see the error numbers go back up as K increased.
KNN and the curse of dimensionality
Note that KNN isn't the best choice for extremely large datasets, and/or models with high dimensionality. This is because the time complexity (what computer scientists call "Big O", which you saw briefly earlier) of this algorithm is exponential. As you add more data points to the dataset, the number of operations needed to complete all the steps of the algorithm grows exponentially! That said, for smaller datasets, KNN often works surprisingly well, given the simplicity of the overall algorithm. However, if your dataset contains millions of rows and thousands of columns, you may want to choose another algorithm, as the algorithm may not run in any reasonable amount of time;in some cases, it could quite literally take years to complete!
Summary
In this lesson you learned how to determine the best value for K and that the KNN algorithm may not necessarily be the best choice for large datasets due to the large amount of time it can take for the algorithm to run.

## KNN with scikit-learn
  
## Introduction
In this lesson, you'll explore how to use scikit-learn's implementation of the K-Nearest Neighbors algorithm. In addition, you'll also learn about best practices for using the algorithm.
## Objectives
You will be able to:
•	List the considerations when fitting a KNN model using scikit-learn
## Why use scikit-learn?
While you've written your own implementation of the KNN algorithm, scikit-learn adds many backend optimizations which can make the algorithm perform faster and more efficiently. Building your own implementation of any machine learning algorithm is a valuable experience, providing great insight into how said algorithm works. However, in general, you should always use professional toolsets such as scikit-learn whenever possible; since their implementations will always be best-in-class, in a way a single developer or data scientist simply can't hope to rival on their own. In the case of KNN, you'll find scikit-learn's implementation to be much more robust and fast, because of optimizations such as caching distances in clever ways under the hood.
## Read the sklearn docs
As a rule of thumb, you should familiarize yourself with any documentation available for any libraries or frameworks you use. scikit-learn provides high-quality documentation. For every algorithm, you'll find a general documentation pageLinks to an external site. which tells you inputs, parameters, outputs, and caveats of any algorithm. In addition, you'll also find very informative User GuidesLinks to an external site. that explain both how the algorithm works, and how to best use it, complete with sample code!
For example, the following image can be found in the scikit-learn user guide for K-Nearest Neighbors, along with an explanation of how different parameters can affect the overall performance of the model.
 ![image](https://github.com/user-attachments/assets/1a5a1e85-5f33-4d7a-9b66-e619a8162a43)

## Best practices
You'll also find that scikit-learn provides robust implementations for additional components of the algorithm implementation process such as evaluation metrics. With that, you can easily evaluate models using precision, accuracy, or recall scores on the fly using built-in functions!
With that, it's important to focus on practical questions when completing the upcoming lab. In particular, try to focus on the following questions:
•	What decisions do I need to make regarding my data? How might these decisions affect overall performance?
•	Which predictors do I need? How can I confirm that I have the right predictors?
•	What parameter values (if any) should I choose for my model? How can I find the optimal value for a given parameter?
•	What metrics will I use to evaluate the performance of my model? Why?
•	How do I know if there's room left for improvement with my model? Are the potential performance gains worth the time needed to reach them?
## A final note
After cleaning, preprocessing, and modeling the data in the next lab, you'll be given the opportunity to iterate on your model.


# KNN with scikit-learn - Lab

## Introduction

In this lab, you'll learn how to use scikit-learn's implementation of a KNN classifier on the classic Titanic dataset from Kaggle!
 

## Objectives

In this lab you will:

- Conduct a parameter search to find the optimal value for K 
- Use a KNN classifier to generate predictions on a real-world dataset 
- Evaluate the performance of a KNN model  


## Getting Started

Start by importing the dataset, stored in the `titanic.csv` file, and previewing it.


```python
# Your code here
# Import pandas and set the standard alias 


# Import the data from 'titanic.csv' and store it in a pandas DataFrame 
raw_df = None

# Print the head of the DataFrame to ensure everything loaded correctly 

```

Great!  Next, you'll perform some preprocessing steps such as removing unnecessary columns and normalizing features.

## Preprocessing the data

Preprocessing is an essential component in any data science pipeline. It's not always the most glamorous task as might be an engaging data visual or impressive neural network, but cleaning and normalizing raw datasets is very essential to produce useful and insightful datasets that form the backbone of all data powered projects. This can include changing column types, as in: 


```python
df['col_name'] = df['col_name'].astype('int')
```
Or extracting subsets of information, such as: 

```python
import re
df['street'] = df['address'].map(lambda x: re.findall('(.*)?\n', x)[0])
```

> **Note:** While outside the scope of this particular lesson, **regular expressions** (mentioned above) are powerful tools for pattern matching! See the [regular expressions official documentation here](https://docs.python.org/3.6/library/re.html). 

Since you've done this before, you should be able to do this quite well yourself without much hand holding by now. In the cells below, complete the following steps:

1. Remove unnecessary columns (`'PassengerId'`, `'Name'`, `'Ticket'`, and `'Cabin'`) 
2. Convert `'Sex'` to a binary encoding, where female is `0` and male is `1` 
3. Detect and deal with any missing values in the dataset:  
    * For `'Age'`, replace missing values with the median age for the dataset  
    * For `'Embarked'`, drop the rows that contain missing values
4. One-hot encode categorical columns such as `'Embarked'` 
5. Store the target column, `'Survived'`, in a separate variable and remove it from the DataFrame  

While we always want to worry about data leakage, which is why we typically perform the split before the preprocessing, for this data set, we'll do some of the preprocessing first. The reason for this is that some of the values of the variables only have a handful of instances, and we want to make sure we don't lose any of them.


```python
# Drop the unnecessary columns
df = None
df.head()
```


```python
# Convert Sex to binary encoding
df['Sex'] = None
df.head()
```


```python
# Find the number of missing values in each column

```


```python
# Impute the missing values in 'Age'
df['Age'] = None
df.isna().sum()
```


```python
# Drop the rows missing values in the 'Embarked' column
df = None
df.isna().sum()
```


```python
# One-hot encode the categorical columns
one_hot_df = None
one_hot_df.head()
```


```python
# Assign the 'Survived' column to labels
labels = None

# Drop the 'Survived' column from one_hot_df

```

## Create training and test sets

Now that you've preprocessed the data, it's time to split it into training and test sets. 

In the cell below:

* Import `train_test_split` from the `sklearn.model_selection` module 
* Use `train_test_split()` to split the data into training and test sets, with a `test_size` of `0.25`. Set the `random_state` to 42 


```python
# Import train_test_split 


# Split the data
X_train, X_test, y_train, y_test = None
```

## Normalizing the data

The final step in your preprocessing efforts for this lab is to **_normalize_** the data. We normalize **after** splitting our data into training and test sets. This is to avoid information "leaking" from our test set into our training set (read more about data leakage [here](https://machinelearningmastery.com/data-leakage-machine-learning/) ). Remember that normalization (also sometimes called **_Standardization_** or **_Scaling_**) means making sure that all of your data is represented at the same scale. The most common way to do this is to convert all numerical values to z-scores. 

Since KNN is a distance-based classifier, if data is in different scales, then larger scaled features have a larger impact on the distance between points.

To scale your data, use `StandardScaler` found in the `sklearn.preprocessing` module. 

In the cell below:

* Import and instantiate `StandardScaler` 
* Use the scaler's `.fit_transform()` method to create a scaled version of the training dataset  
* Use the scaler's `.transform()` method to create a scaled version of the test dataset  
* The result returned by `.fit_transform()` and `.transform()` methods will be numpy arrays, not a pandas DataFrame. Create a new pandas DataFrame out of this object called `scaled_df`. To set the column names back to their original state, set the `columns` parameter to `one_hot_df.columns` 
* Print the head of `scaled_df` to ensure everything worked correctly 


```python
# Import StandardScaler


# Instantiate StandardScaler
scaler = None

# Transform the training and test sets
scaled_data_train = None
scaled_data_test = None

# Convert into a DataFrame
scaled_df_train = None
scaled_df_train.head()
```

You may have noticed that the scaler also scaled our binary/one-hot encoded columns, too! Although it doesn't look as pretty, this has no negative effect on the model. Each 1 and 0 have been replaced with corresponding decimal values, but each binary column still only contains 2 values, meaning the overall information content of each column has not changed.

## Fit a KNN model

Now that you've preprocessed the data it's time to train a KNN classifier and validate its accuracy. 

In the cells below:

* Import `KNeighborsClassifier` from the `sklearn.neighbors` module 
* Instantiate the classifier. For now, you can just use the default parameters  
* Fit the classifier to the training data/labels
* Use the classifier to generate predictions on the test data. Store these predictions inside the variable `test_preds` 


```python
# Import KNeighborsClassifier


# Instantiate KNeighborsClassifier
clf = None

# Fit the classifier


# Predict on the test set
test_preds = None
```

## Evaluate the model

Now, in the cells below, import all the necessary evaluation metrics from `sklearn.metrics` and complete the `print_metrics()` function so that it prints out **_Precision, Recall, Accuracy, and F1-Score_** when given a set of `labels` (the true values) and `preds` (the models predictions). 

Finally, use `print_metrics()` to print the evaluation metrics for the test predictions stored in `test_preds`, and the corresponding labels in `y_test`. 


```python
# Your code here 
# Import the necessary functions

```


```python
# Complete the function
def print_metrics(labels, preds):
    print("Precision Score: {}".format(None))
    print("Recall Score: {}".format(None))
    print("Accuracy Score: {}".format(None))
    print("F1 Score: {}".format(None))
    
print_metrics(y_test, test_preds)
```

> Interpret each of the metrics above, and explain what they tell you about your model's capabilities. If you had to pick one score to best describe the performance of the model, which would you choose? Explain your answer.

Write your answer below this line: 


________________________________________________________________________________




## Improve model performance

While your overall model results should be better than random chance, they're probably mediocre at best given that you haven't tuned the model yet. For the remainder of this notebook, you'll focus on improving your model's performance. Remember that modeling is an **_iterative process_**, and developing a baseline out of the box model such as the one above is always a good start. 

First, try to find the optimal number of neighbors to use for the classifier. To do this, complete the `find_best_k()` function below to iterate over multiple values of K and find the value of K that returns the best overall performance. 

The function takes in six arguments:
* `X_train`
* `y_train`
* `X_test`
* `y_test`
* `min_k` (default is 1)
* `max_k` (default is 25)
    
> **Pseudocode Hint**:
1. Create two variables, `best_k` and `best_score`
1. Iterate through every **_odd number_** between `min_k` and `max_k + 1`. 
    1. For each iteration:
        1. Create a new `KNN` classifier, and set the `n_neighbors` parameter to the current value for k, as determined by the loop 
        1. Fit this classifier to the training data 
        1. Generate predictions for `X_test` using the fitted classifier 
        1. Calculate the **_F1-score_** for these predictions 
        1. Compare this F1-score to `best_score`. If better, update `best_score` and `best_k` 
1. Once all iterations are complete, print the best value for k and the F1-score it achieved 


```python
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    # Your code here
    pass

```


```python
find_best_k(scaled_data_train, y_train, scaled_data_test, y_test)
# Expected Output:

# Best Value for k: 17
# F1-Score: 0.7468354430379746
```

If all went well, you'll notice that model performance has improved by 3 percent by finding an optimal value for k. For further tuning, you can use scikit-learn's built-in `GridSearch()` to perform a similar exhaustive check of hyperparameter combinations and fine tune model performance. For a full list of model parameters, see the [sklearn documentation !](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)



## (Optional) Level Up: Iterating on the data

As an optional (but recommended!) exercise, think about the decisions you made during the preprocessing steps that could have affected the overall model performance. For instance, you were asked to replace the missing age values with the column median. Could this have affected the overall performance? How might the model have fared if you had just dropped those rows, instead of using the column median? What if you reduced the data's dimensionality by ignoring some less important columns altogether?

In the cells below, revisit your preprocessing stage and see if you can improve the overall results of the classifier by doing things differently. Consider dropping certain columns, dealing with missing values differently, or using an alternative scaling function. Then see how these different preprocessing techniques affect the performance of the model. Remember that the `find_best_k()` function handles all of the fitting; use this to iterate quickly as you try different strategies for dealing with data preprocessing! 


```python

```


```python

```


```python

```


```python

```

## Summary

Well done! In this lab, you worked with the classic Titanic dataset and practiced fitting and tuning KNN classification models using scikit-learn! As always, this gave you another opportunity to continue practicing your data wrangling skills and model tuning skills using Pandas and scikit-learn!
