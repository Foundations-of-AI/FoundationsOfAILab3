#### Lab 3a. EDA. In this lab, we'll look at a dataset of trees in San Francisco
### and learn how to handle some common cleaning issues.

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 40)
plt.rcParams['figure.figsize'] = (8, 5)

### To start, let's read in the dataset and see what it looks like.

df = pd.read_csv('San_Francisco_Street_Tree_Inventory_20260921.csv')
print(df.shape)
df.head()

### We're getting a warning that some of the columns have mixed data types.
### Let's check that out.

## Question 1: Look at the siteorder column with

df['siteorder'].apply(type).value_counts().

## What two types is it mixing, and why do you think that happened?

### Dealing with missing data is a common problem. First, we need to see what is missing.

## isna creates a mask of the dataframe with True/1 if they are missing and False/0 otherwise.
## mean() then computes the average of the masked cells, which winds up being the percentage of missing data

missing_pct = df.isna().mean().sort_values(ascending=False)
missing_pct.round(3)

### Question 2. Which features have the highest percentage of missing data?

### One strategy is to drop rows with missing data. You can do this with

df.dropna()

## Question 3. How much data would we lose if we did this?


## If we don't want to drop the rows, we could instead add synthetic data. This is called
# imputing. There's a few options:

## We could use pre-determined constant values. We can pick different ones for each column:
## This is helpful for categorical or string data, where we want to know that a value is synthetic.

default_vals = {"planteddate":"1999-12-31",  'legalstatus':"DPW Maintained", 'mapdbh':3 }

filled_data = df.fillna(default_vals)

## We could use the mean, or average value (for numeric variables)
## This is smart if our data is normally distributed. If it is skewed, we might get into trouble.

filled_data['mapdbh'] = df['mapdbh'].fillna(df['mapdbh'].mean())

## We could also use the mode, which is the most common value.
## This is nice for categorical data. For numerical data, it can introduce skewing.

filled_data['mapdbh'] = df['mapdbh'].fillna(df['mapdbh'].mode()[0])

## or we could ask pandas to fill in the data based on the other data values.

filled_data['mapdbh'] = df['mapdbh'].interpolate(method='linear')

#Question 4: For the 'siteinfo', 'legalstatus', 'xcoord', and 'ycoord' elements, which strategy
# would you suggest to deal with missing data? Does the fact that 'xcoord' and 'ycoord' are related
# change your decision?


### Question 5. 'planteddate' is missing many of its entries. Let's consider three possible explanations:
## 1. Some of the planters neglected to add this data
## 2. Older trees are missing this data; it wasn't recorded initially.
## 3. This is random; there's no particular pattern.
## How would you test each of these hypotheses? You don't need to actually do the test, but write down an
## explanation for what you would explore.





## The 'species' column is kind of messy. Let's take a look at it.

df['species'].dropna().head(5).tolist()

## The scientific and common names are combined into one field. Let's fix this.

## This creates a new dataframe with two columns
split_cols = df['species'].str.split(' :: ', n=1, expand=True)

## Now we add new columns to our original dataframe.
df['scientific_name'] = split_cols[0]
df['common_name'] = split_cols[1]
df[['species', 'scientific_name', 'common_name']].head()

##Question 6: So, did this work? How can you check to see whether all of the rows were split correctly without
## searching through all of the data by hand? (Hint: str.contains() will tell you whether a pattern
## is present. How could we determine whether a pattern is absent?

### Question 6: Let's take a look at 'siteinfo'

df['siteinfo'].head(5)

# It has a similar issue. There are three fields combined together. Implement a solution that breaks
# this into separate columns, as we did above.

## Question 7. Now that you've had a chance to look at this data a little bit, what is a question
## that you think would be interesting to investigate with it? You don't need to implement this,
## but please provide a few sentences describing the question you would want to ask, and how you would
## evaluate this.

## For example: I heard that my supervisor in District 9 has done a really good job of promoting
## tree planting. If that's true, there should be significantly more trees in district 9 than in
## other districts. I would do this by counting the number of trees in each district, filtering to
## only include rows with 'planteddate' within the dates of their term in office. I'll need to think
## about how to deal with the missing data for 'planteddate'.

