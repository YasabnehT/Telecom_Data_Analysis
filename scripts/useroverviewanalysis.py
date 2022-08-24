# -*- coding: utf-8 -*-
"""UserOverviewAnalysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I3mDv95t0u0HQA_svNcy24Kh3xxTHwCx

## Module Imports
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns

"""### Mount Google Drive to Google Colab"""

from google.colab import drive
drive.mount('/content/drive')

"""# Data Understanding

### **Identify datasets with NaN or None values**
"""

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_column', None)
db = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/Week1_challenge_data_source(CSV).csv', na_values=['undefined','?', None])
db.head() # the fisrt five rows

"""# Size of the dataset
### Columns of the dataset
"""

# list of column  names
db.columns.tolist()

"""### Number of columns"""

print(f"Number of columns: ", len(db.columns))

"""### Number of data points and data size"""

print(f" There are {db.shape[0]} rows and {db.shape[1]} columns")

"""### Features/columns and their data type"""

db.dtypes

"""

```
# This is formatted as code
```

### Utility Functions"""

# how many missing values exist or better still what is the % of missing values in the dataset?
def percent_missing(df):

    # Calculate total number of cells in dataframe
    totalCells = np.product(df.shape)

    # Count number of missing values per column
    missingCount = df.isnull().sum()

    # Calculate total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    print("The dataset contains", round(((totalMissing/totalCells) * 100), 3), "%", "missing values.")

percent_missing(db)

# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def format_float(value):
    return f'{value:,.2f}'

def find_agg(df:pd.DataFrame, agg_column:str, agg_metric:str, col_name:str, top:int, order=False )->pd.DataFrame:
    
    new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name).\
                        sort_values(by=col_name, ascending=order)[:top]
    
    return new_df

def convert_bytes_to_megabytes(df, bytes_data):
    """
        This function takes the dataframe and the column which has the bytes values
        returns the megabytesof that value
        
        Args:
        -----
        df: dataframe
        bytes_data: column with bytes values
        
        Returns:
        --------
        A series
    """
    
    megabyte = 1*10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]

def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(),df[column])
    
    return df[column]


###################################PLOTTING FUNCTIONS###################################

def plot_hist(df:pd.DataFrame, column:str, color:str)->None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def plot_count(df:pd.DataFrame, column:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x=column)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()
    
def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()

def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    plt.show()

def plot_box(df:pd.DataFrame, x_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()

def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()


pd.options.display.float_format = format_float

"""### Missing Value table"""

missing_values_table(db)

"""### Columns with missing values count

The Column "Nb of sec with 37500B < Vol UL" has maximum missing values of 130254 occurances
"""

db.isna().sum() # missing values of each column

print ("Maximum missing values per column: ", np.max(db.isna().sum())) # print(db.isna().sum().max())

"""### Maximum values of each column"""

db.max()

"""### Minimum values of each column"""

db.min()

"""### Top 10 Handsets used"""

db_hndset_count = db['Handset Type'].value_counts()
top_10_hndsets = db_hndset_count.head(10)
print("Most used handset types in Descending order:\n", db_hndset_count)
print("\n\nTop 10 handsets used: \n", top_10_hndsets)

"""### Top 3 handset manufacturers"""

db_hndset_manufac_count = db['Handset Manufacturer'].value_counts()
top_3_manufact = db_hndset_manufac_count.head(3)
print("Dominant manufacturers in descending order:\n", db_hndset_manufac_count)
print("\n\nTop 3 manufacturers: \n", top_3_manufact)

"""### Manufacturer-Handset pairs"""

db_hndset_manufac_pair = db.value_counts(["Handset Manufacturer", "Handset Type"])
top_3_manufact_5_hndset = db_hndset_manufac_pair.head(3)
print("Manufacturers-handset pair:\n", top_3_manufact_5_hndset)

"""### Data Aggregation with each column"""

db['Bearer Id'].value_counts() # Each xDR occurances aggregated
# db.value_counts('Bearer Id') also works

"""### User (MSISDN) Grouped and Agregated with Bearer Id(xDR session)
Each user has unique xDR session
"""

# db_user_xDR = db.groupby(["IMEI","Bearer Id"]).agg(session_count = ('Bearer Id', 'value_counts')) # it also works
db_user_xDR = db.groupby(["MSISDN/Number","Bearer Id"]).size()
db_user_xDR

"""### User(MSISDN) Grouped and Aggregated with xDR duration"""

db_user_Duration = db.groupby(["MSISDN/Number","Dur. (ms)"]).size() #transform(sum)
db_user_Duration

"""### User(MSISDN) and Total UL(Upload) Grouped and Aggregated"""

db_user_UL_data = db.groupby(["MSISDN/Number","Total UL (Bytes)"]).size()
db_user_UL_data

"""### User(MSISDN) and total download(DL) grouped and aggregated"""

db_user_DL_data = db.groupby(["MSISDN/Number","Total DL (Bytes)"]).size()
db_user_DL_data

"""### User (MSISDN) aggregated with Social Media DL data volume"""

db_user_DL_social_media = db.groupby(["MSISDN/Number","Social Media DL (Bytes)"]).size()
db_user_DL_social_media

"""### Data volume for Social Media UL (Bytes)"""

db_user_UL_social_media = db.groupby(["IMEI","Social Media UL (Bytes)"]).size()
db_user_UL_social_media

"""### Data volume for YouTube DL (Bytes)"""

db_user_DL_Youtube = db.groupby(["IMEI","Youtube DL (Bytes)"]).size()
db_user_DL_Youtube

"""### Data volume for YouTube UL (Bytes)"""

db_user_UL_Youtube = db.groupby(["IMEI","Youtube UL (Bytes)"]).size()
db_user_UL_Youtube

"""### Data volume for Netflix DL (Bytes)"""

db_user_DL_Netflix = db.groupby(["IMEI","Netflix DL (Bytes)"]).size()
db_user_DL_Netflix

"""### Data volume for Netflix UL (Bytes)"""

db_user_UL_Netflix = db.groupby(["IMEI","Netflix UL (Bytes)"]).size()
db_user_UL_Netflix

"""### Data volume for Google DL (Bytes)"""

db_user_DL_Google = db.groupby(["IMEI","Google DL (Bytes)"]).size()
db_user_DL_Google

"""### Data volume for Google UL (Bytes)"""

db_user_UL_Google = db.groupby(["IMEI","Google UL (Bytes)"]).size()
db_user_UL_Google

"""### Data volume for Email DL (Bytes)"""

db_user_DL_Email = db.groupby(["IMEI","Email DL (Bytes)"]).size()
db_user_DL_Email

"""### Data volume for Email UL (Bytes)"""

db_user_UL_Email = db.groupby(["IMEI","Email UL (Bytes)"]).size()
db_user_UL_Email

"""### Data volume for Gaming DL (Bytes)"""

# db_user_DL_Gaming = db.groupby(["IMEI","Gaming DL (Bytes)"]).agg({'Gaming DL (Bytes)':'sum'})#.size()
db_user_DL_Gaming = db.groupby(["IMEI"]).agg({'Gaming DL (Bytes)':'sum'})#.size()

db_user_DL_Gaming

"""### Data volume for Gaming UL (Bytes)"""

db_user_UL_Gaming = db.groupby(["IMEI","Gaming UL (Bytes)"]).size()
db_user_UL_Gaming

"""### Data volume for Other DL"""

db_user_DL_Other = db.groupby(["IMEI","Other DL (Bytes)"]).size()
db_user_DL_Other

"""### Data Volume for Other UL"""

db_user_UL_Other = db.groupby(["IMEI","Other UL (Bytes)"]).size()
db_user_UL_Other

"""# Data Exploration

Use Mode method to fill the missing datapoints of all 'object' type features and Mean/Median methods for all numuric type features.
*   use Median method for skewed(negative/positive) numeric feature and 
*   use MEAN/Median for non-skewd/symetrical numeric feature

### Method selection based on data skewness

#### Skewness of each column
"""

db.skew(axis=0)

"""### Skewness visualization with histogram"""

db['Total UL (Bytes)'].hist() #skewness of Total upload column

db['Total DL (Bytes)'].hist()

db['Total UL (Bytes)'].hist()

"""### Positively skewed parameter"""

db['HTTP DL (Bytes)'].hist()

"""### Negatively skewwed parameter

"""

db['UL TP < 10 Kbps (%)'].hist()

"""### Data with total missing values in each column - revisited"""

db.isna().sum()

"""### Column data types - revisited"""

db.dtypes

"""### Utility function to fill missing values
* numeric missing values with mean method
* object type missing values with mode method
"""

# fill numeric columns with ffill and bfill
"""
df[col].fillna(method='ffill') and df[col].fillna(method='bfill') or 
df[col].ffill(axis = 0) and df[col].bfill(axis = 0) fills the missing values with the value before/after it
"""
# fill missing numeric values with mean and object type values with mode
def fill_missing_values(df):
  for column in df.columns:
    if df[column].dtype == 'float64':
      df[column] = df[column].fillna(df[column].mean())
    elif df[column].dtypes == 'object':
      df[column] = df[column].fillna(df[column].mode()[0])
  return df

"""### Data with all missing values filled - zero null count


"""

fill_missing_values(db).isna().sum()

"""### Other method of handling missing values - Interpolation
* We can use interpolation while working with time-series data because in time-series data we like to fill missing values with previous one or two values.
* It can be used to estimate unknown data points between two known data points.

##### Since we are not considering the time-series nature of the telecom data, we choose not to use interpolation here.
"""

# db.interpolate(inplace=True)

"""## Data Transformation

**Scaling and Normalization**

##### Scaling - changing the range of your data 
##### Normalization, you're changing the shape of the distribution of your data.

#### Scaling

* This transforms data so that it fits within a specific scale, like 0-100 or 0-1. 
* It is important when we're using methods based on distance measures of data points like support vector machines (SVM) or k-nearest neighbors (KNN).
* We use the scaler method from sklearn.

#### Normalization

Scaling just changes the range of your data. Normalization is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution.

* Normal distribution ("bell curve", Gaussian distribution) is a specific statistical distribution where a roughly equal observations fall above and below the mean
 * The mean and the median are the same, and there are more observations closer to the mean.

* In general, you'll normalize your data if you're going to be using a machine learning or statistics technique like LDA and Gaussian naive Bayes that assumes your data is normally distributed. Some examples of these include linear discriminant analysis (LDA) and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in the name probably assumes normality.)

* We usee the Normalizer method from sklearn

### Numeric value scalling
"""

minmax_scaler = preprocessing.MinMaxScaler()
def scalling_numeric_values(df):
  col_values = []
  for column in df.columns:
    if df[column].dtype == 'float64':
      col_values.append(list(df[column].values))
  col_values_skaled = minmax_scaler.fit_transform(col_values)
  db_scaled = pd.DataFrame(col_values_skaled)
  return df

db_sklearn = fill_missing_values(db.copy())
scalling_numeric_values(db_sklearn)

"""### Scaling between [0,1]"""

def scalling_numeric_values_0_1(df):
  for column in df.columns:
    if df[column].dtype == 'float64':
      df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(150001,1))
  return df

db_sklearn = fill_missing_values(db.copy())
scalling_numeric_values_0_1(db_sklearn)

"""### Min values in each column"""

db.min()

"""### Max values in each column"""

db.max()

"""### Data Extraction"""

db['MSISDN/Number'].value_counts()

db['Dur. (ms)'].value_counts()

percent_missing(db)

missing_values_table(db)

"""### Mean and Mediam of some vital attributes"""

important_columns_numeric = ['Bearer Id','Dur. (ms)','MSISDN/Number',
                      'Avg RTT DL (ms)','Avg RTT UL (ms)',
                      'TCP DL Retrans. Vol (Bytes)','TCP UL Retrans. Vol (Bytes)',
                      'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                      'Google DL (Bytes)', 'Google UL (Bytes)', 
                      'Email DL (Bytes)','Email UL (Bytes)',
                      'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                      'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                      'Gaming DL (Bytes)','Gaming UL (Bytes)',
                      'Other DL (Bytes)', 'Other UL (Bytes)',
                      'Total UL (Bytes)', 'Total DL (Bytes)' ]

important_columns_object = ['Handset Manufacturer','Handset Type']
db[important_columns_numeric].mean() #mean of numeric columns

db[important_columns_numeric].median()

db[important_columns_object].mode()

"""### Univariate analysis - Analysis using only one feature/variable"""

db_explore = db.copy()

fix_outlier(db_explore, "MSISDN/Number")

plot_hist(db_explore.head(10000),"MSISDN/Number" ,'green')

plot_hist(db_explore, "Dur. (ms)", "green")

plot_hist(db_explore, "Bearer Id", "green")

plot_hist(db_explore, "Avg RTT DL (ms)", "green")

plot_hist(db_explore, "Avg RTT UL (ms)", "green")

plot_hist(db_explore.head(50000), "Handset Manufacturer", "blue")

plot_hist(db_explore.head(50000), "Handset Type", "blue")

plot_hist(db_explore, "Social Media DL (Bytes)", "green")
# sns.histplot(x=columns[0], data =db) # this also works

plot_hist(db_explore, "Social Media UL (Bytes)", "green")

plot_hist(db_explore, "Total DL (Bytes)", "green")

plot_hist(db_explore, "Total UL (Bytes)", "green")

plot_box(db_explore, "Dur. (ms)", "Session Duration Outliers")

plot_box(db_explore, "Avg RTT DL (ms)", "Avg RTT DL (ms) Outliers")

plot_box(db_explore, "Avg RTT UL (ms)", "Avg RTT UL (ms) Outliers")

plot_box(db_explore, "TCP DL Retrans. Vol (Bytes)", "TCP DL Retrans. Vol (Bytes) Outliers")

plot_box(db_explore, "TCP UL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes) Outliers")

plot_box(db_explore, "Social Media DL (Bytes)", "Social Media DL (Bytes) Outliers")

plot_box(db_explore, "Social Media UL (Bytes)", "Social Media UL (Bytes) Outliers")

plot_box(db_explore, "Total DL (Bytes)", "Total DL (Bytes) Outliers")

plot_box(db_explore, "Total UL (Bytes)", "Total UL (Bytes) Outliers")

"""### Categorical Data Plot"""

plot_count(db_explore, "Handset Manufacturer")

plot_count(db_explore, "Handset Type")

"""### Non-graphical Univariat EDA"""

db.describe()

db["Total DL (Bytes)"].describe()

db["Total UL (Bytes)"].describe()

db["MSISDN/Number"].describe()

db.info()

db.isna().sum()

# sns.histplot(x=db[columns]['Total UL (Bytes)'], data =db)

# sns.histplot(x=db[columns]['Total DL (Bytes)'], data =db)

"""### Bivariate analysis
#### Applications Vs Total DL and Total UL
"""

db_explore_100 = db_explore.head(100)
sns.barplot(x='Total DL (Bytes)',y='Social Media DL (Bytes)',data=db_explore_100)
# sns.countplot(x='Total DL (Bytes)',data=db) 
#boxplot, violinplot, stripplot, swarmplot, barplot also works

sns.barplot(x='Total DL (Bytes)',y='Social Media UL (Bytes)',data=db_explore.head(1000))

sns.barplot(x='Total DL (Bytes)',y='Social Media UL (Bytes)',data=db_explore_100)

sns.barplot(x='Total UL (Bytes)',y='Social Media DL (Bytes)',data=db_explore_100)

sns.barplot(x='Total UL (Bytes)',y='Social Media UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Google DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Google UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Google DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Google UL (Bytes)',data=db_explore_100)

sns.stripplot(x='Total DL (Bytes)',y='Email DL (Bytes)',data=db_explore_100)

sns.stripplot(x='Total DL (Bytes)',y='Email UL (Bytes)',data=db_explore_100)

sns.stripplot(x='Total UL (Bytes)',y='Email DL (Bytes)',data=db_explore_100)

sns.stripplot(x='Total UL (Bytes)',y='Email UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Youtube DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Youtube UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Youtube DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Youtube UL (Bytes)',data=db_explore_100)

sns.barplot(x='Total DL (Bytes)',y='Netflix DL (Bytes)',data=db_explore_100)

sns.barplot(x='Total DL (Bytes)',y='Netflix UL (Bytes)',data=db_explore_100)

sns.barplot(x='Total UL (Bytes)',y='Netflix DL (Bytes)',data=db_explore_100)

sns.barplot(x='Total UL (Bytes)',y='Netflix UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Gaming DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Gaming UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Gaming DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Gaming UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Other DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total DL (Bytes)',y='Other UL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Other DL (Bytes)',data=db_explore_100)

sns.regplot(x='Total UL (Bytes)',y='Other UL (Bytes)',data=db_explore_100)

"""### Multivariate Analysis"""

plot_scatter(db_explore.head(100), x_col="MSISDN/Number", y_col="Social Media DL (Bytes)", hue="Social Media UL (Bytes)",
             style="Social Media UL (Bytes)", title="Social media DL consumption per user")

plot_scatter(db_explore.head(100), x_col="MSISDN/Number", y_col="Total DL (Bytes)", hue="Total UL (Bytes)",
             style="Total UL (Bytes)", title="Total DL consumption per user")

plot_box_multi(db_explore.head(100), x_col="MSISDN/Number", y_col="TCP DL Retrans. Vol (Bytes)", 
               title="TCP DL Retrans. Vol (Bytes) outilers in MSISDN/Number column")

dfPair = db_explore.head(50)[["MSISDN/Number", "Dur. (ms)", "Avg RTT DL (ms)", "Social Media DL (Bytes)", "Total DL (Bytes)"]]
sns.pairplot(dfPair, hue = 'Total DL (Bytes)', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             height=4)

dfPair = db_explore.head(50)[["MSISDN/Number", "Dur. (ms)", "Avg RTT DL (ms)", "Social Media DL (Bytes)", "Total DL (Bytes)"]]
sns.pairplot(dfPair, hue = 'Total DL (Bytes)', diag_kind = 'kde',height=4)

"""### Deciles

### Selected columns based on separate PCA
"""

decile_columns = ['MSISDN/Number','Dur. (ms)','Total UL (Bytes)', 'Total DL (Bytes)' ] # to limit the number of columns to be displayed
db_decile = db_explore[decile_columns]
# db_decile_group["Dur. Decile"] = pd.qcut(db_decile_group['Dur. (ms)'], 5, labels = ['Dec 1','Dec 2','Dec 3','Dec 4','Dec 5'])
# db_decile_group

"""### Five MSISDN deciles based on xDR Duration
#### contains all selected columns
"""

db_decile_group_dur = db_decile.groupby(pd.qcut(db_decile["Dur. (ms)"], 5))
db_decile_group_dur.describe() # includes all selected columns

"""### Deciles based on xDR duration
### Contains only the xDR duration data
"""

db_decile_group_dur['Dur. (ms)'].describe()

"""### Decile Total DL Bytes sum"""

db_decile_group_dur['Total DL (Bytes)'].sum()

"""### Decile Total UL Bytes sum"""

db_decile_group_dur['Total UL (Bytes)'].sum()

"""### Correlation Analysis

### Correlation Analysis for the whole data
"""

db.corr(method='pearson')

"""### Correlation Analysis for individual columns
#### Can be calculated using 'pearson’, ‘kendall’, ‘spearman methods; pearson being the standard correlation coefficient
"""

cor_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
               'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
               'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)'] 


db[cor_columns].corr(method='pearson')

"""### Unapproximated correlation pairwise coefficients"""

db[cor_columns[0]].corr(db[cor_columns[1]], method = 'pearson')

def Iterative_corr():
  for i in range(0,len(cor_columns)):
    print(f"Correlation between {cor_columns[i]} and {cor_columns[i+1]} is {db[cor_columns[i]].corr(db[cor_columns[i+1]], method = 'pearson')}")

Iterative_corr()

