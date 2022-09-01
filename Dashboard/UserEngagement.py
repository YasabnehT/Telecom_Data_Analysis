import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import plotly.express as px
import matplotlib 

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

from sklearn.model_selection import train_test_split
from joblib import dump,load
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import scipy.spatial.distance as sdist
import pickle

def fill_missing_values(df):
  for column in df.columns:
    if df[column].dtype == 'float64':
      df[column] = df[column].fillna(df[column].mean())
    elif df[column].dtypes == 'object':
      df[column] = df[column].fillna(df[column].mode()[0])
  return df


@st.cache()
def load_data():
    db = pd.read_csv("data/db_explore.csv")
    return fill_missing_values(db)

db = load_data()


"""# User Engagement Analysis

### Top 10 Handsets used
"""

db_hndset_count = db['Handset Type'].value_counts()
top_10_hndsets = db_hndset_count.head(10)
db_hndset_count 
db_hndset_count.head(10)

"""### Top 3 handset manufacturers"""

db_man =db['Handset Manufacturer'].value_counts()
db_man.head(3)


"""## Data Aggregation with each column

### # Most Frequent Sessions
"""

db_session = db['Bearer Id'].value_counts()
db_session.head(3)
"""### Frequency of each User(MSISDN/Number)"""

db["MSISDN/Number"].value_counts()

"""### Top 10 frequent users"""

db["MSISDN/Number"].value_counts().head(10)


"""### Top 10 frequent users-session pairs"""

db_user_xDR = db.groupby(["MSISDN/Number"]).agg(session_count = ('Bearer Id', 'value_counts')).sort_values(by='session_count', ascending = False)#.value_counts(ascending = False) # it also works
db_user_xDR.head(10)

"""### User(MSISDN) Grouped and Aggregated with xDR duration"""

"""### Top 10 frequent durations users have engaged
#### These are frequent durations when users stay connected
"""

db_user_Duration = db.groupby(["MSISDN/Number"]).agg(Duration = ('Dur. (ms)', 'value_counts')).sort_values(by='Duration', ascending = False)#.value_counts(ascending = False) # it also works

db_user_Duration.head(10)

"""### Top 10 user with longer engagement durations (ms)
#### These are users who engaged more by staying connected longer
"""

db_user_Duration = db.groupby(["MSISDN/Number"]).agg(Duration_ms = ('Dur. (ms)', 'sum')).sort_values(by='Duration_ms', ascending = False)#.value_counts(ascending = False) # it also works

db_user_Duration.head(10)

"""## User-Total Bytes Aggregation"""

# Total Bytes (Total UL + Total DL)
db_Total_Bytes = db.copy()
db_Total_Bytes["Total Bytes"] = db["Total DL (Bytes)"] + db["Total UL (Bytes)"]

# Total Social Media Bytes
db_Total_Bytes["Total Social Media (Bytes)"] = db["Social Media DL (Bytes)"] + db["Social Media UL (Bytes)"]

#Total Google Bytes
db_Total_Bytes["Total Google (Bytes)"] = db["Google DL (Bytes)"] + db["Google UL (Bytes)"]

#Total Youtube Bytes
db_Total_Bytes["Total Youtube (Bytes)"] = db["Youtube DL (Bytes)"] + db["Youtube UL (Bytes)"]

#Total Email Bytes
db_Total_Bytes["Total Email (Bytes)"] = db["Email DL (Bytes)"] + db["Email UL (Bytes)"]

#Total Netflix Bytes
db_Total_Bytes["Total Netflix (Bytes)"] = db["Netflix DL (Bytes)"] + db["Netflix UL (Bytes)"]

#Total Gaming Bytes
db_Total_Bytes["Total Gaming (Bytes)"] = db["Gaming DL (Bytes)"] + db["Gaming UL (Bytes)"]

#Total Other Bytes
db_Total_Bytes["Total Other (Bytes)"] = db["Other DL (Bytes)"] + db["Other UL (Bytes)"]


### User Experience ######
# Total TCP Retransmission Bytes
db_Total_Bytes["Total TCP Retrans. (Bytes)"] = db['TCP DL Retrans. Vol (Bytes)'] + db['TCP UL Retrans. Vol (Bytes)']

# Total Avg RTT 
db_Total_Bytes["Total Avg RTT (ms)"] = db['Avg RTT DL (ms)'] + db['Avg RTT UL (ms)']

# Total Avg Bearer TP
db_Total_Bytes["Total Avg TP (kbps)"] = db['Avg Bearer TP DL (kbps)'] + db['Avg Bearer TP DL (kbps)']

db_totals_col = db_Total_Bytes[["MSISDN/Number",'Dur. (ms)', "Handset Type", "Total Bytes", "Total TCP Retrans. (Bytes)", 
                "Total Avg RTT (ms)", "Total Avg TP (kbps)","Total Social Media (Bytes)", 
               "Total Google (Bytes)", "Total Youtube (Bytes)","Total Email (Bytes)", 
               "Total Netflix (Bytes)", "Total Gaming (Bytes)", "Total Other (Bytes)"]]
db_totals_col.head()

"""### User-Total Social Media Bytes

"""

db_user_Social = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Social_Media_Bytes = ("Total Social Media (Bytes)", 'sum')).sort_values(by='Total_Social_Media_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Social

"""### Top 10 Social Media Users"""

db_user_Social.head(10)

"""### User-Total Google Bytes"""

db_user_Google = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Google_Bytes = ("Total Google (Bytes)", 'sum')).sort_values(by='Total_Google_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Google

"""### Top 10 Google Users"""

db_user_Google.head(10)

"""### User-Total Youtube Bytes"""

db_user_Youtube = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Youtube_Bytes = ("Total Youtube (Bytes)", 'sum')).sort_values(by='Total_Youtube_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Youtube

"""### Top 10 Youtube Users"""

db_user_Youtube.head(10)

"""### User-Total Email Bytes"""

db_user_Email = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Email_Bytes = ("Total Email (Bytes)", 'sum')).sort_values(by='Total_Email_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Email

"""### Top 10 Email Users"""

db_user_Email.head(10)

"""### User-Total Netflix Bytes"""

db_user_Netflix = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Netflix_Bytes = ("Total Netflix (Bytes)", 'sum')).sort_values(by='Total_Netflix_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Netflix

"""### Top 10 Netflix Users"""

db_user_Netflix.head(10)

"""### User-Total Gaming Bytes"""

db_user_Gaming = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Gaming_Bytes = ("Total Gaming (Bytes)", 'sum')).sort_values(by='Total_Gaming_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Gaming

"""### Top 10 Gaming Users"""

db_user_Gaming.head(10)

"""### User-Total Other Services Bytes"""

db_user_Other = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Other_Bytes = ("Total Other (Bytes)", 'sum')).sort_values(by='Total_Other_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Other

"""### Top 10 Other Services Users"""

db_user_Other.head(10)

"""### User (MSISDN) aggregated with Social Media DL data volume"""

db_user_social_DL = db.groupby(["MSISDN/Number"]).agg(Social_Media_DL_Bytes = ("Social Media UL (Bytes)", 'sum')).sort_values(by='Social_Media_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_social_DL

"""### Top 10 users with largest Social Media Download"""

db_user_social_DL.head(10)

"""### Data volume for Social Media UL (Bytes)"""

db_user_social_UL = db.groupby(["MSISDN/Number"]).agg(Social_Media_UL_Bytes = ("Social Media UL (Bytes)", 'sum')).sort_values(by='Social_Media_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_social_UL

"""### Top 10 users with lasrgest Social Media Upload"""

db_user_social_UL.head(10)

"""### Data volume for YouTube DL (Bytes)"""

db_user_youtube_DL = db.groupby(["MSISDN/Number"]).agg(Youtube_DL_Bytes = ("Youtube DL (Bytes)", 'sum')).sort_values(by='Youtube_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_youtube_DL

"""### Top 10 users with largest Youtube Download"""

db_user_youtube_DL.head(10)

"""### Data volume for YouTube UL (Bytes)"""

db_user_youtube_UL = db.groupby(["MSISDN/Number"]).agg(Youtube_UL_Bytes = ("Youtube UL (Bytes)", 'sum')).sort_values(by='Youtube_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_youtube_UL

"""### Top 10 users with largest Youtube Upload"""

db_user_youtube_UL.head(10)

"""### Data volume for Netflix DL (Bytes)"""

db_user_Netflix_DL = db.groupby(["MSISDN/Number"]).agg(Netflix_DL_Bytes = ("Netflix DL (Bytes)", 'sum')).sort_values(by='Netflix_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Netflix_DL

"""### Top 10 users with largest Netflix Download"""

db_user_Netflix_DL.head(10)

"""### Data volume for Netflix UL (Bytes)"""

db_user_Netflix_UL = db.groupby(["MSISDN/Number"]).agg(Netflix_UL_Bytes = ("Netflix UL (Bytes)", 'sum')).sort_values(by='Netflix_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Netflix_UL

"""### Top 10 users with Netflix Upload"""

db_user_Netflix_UL.head(10)

"""### Data volume for Google DL (Bytes)"""

db_user_Google_DL = db.groupby(["MSISDN/Number"]).agg(Google_DL_Bytes = ("Google DL (Bytes)", 'sum')).sort_values(by='Google_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Google_DL

"""### Top 10 users with largest Google Download"""

db_user_Google_DL.head(10)

"""### Data volume for Google UL (Bytes)"""

db_user_Google_UL = db.groupby(["MSISDN/Number"]).agg(Google_UL_Bytes = ("Google UL (Bytes)", 'sum')).sort_values(by='Google_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Google_UL

"""### Top 10 users with largest Google Upload"""

db_user_Google_UL.head(10)

"""### Data volume for Email DL (Bytes)"""

db_user_Email_DL = db.groupby(["MSISDN/Number"]).agg(Email_DL_Bytes = ("Email DL (Bytes)", 'sum')).sort_values(by='Email_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Email_DL

"""### Top 10 users with largest Email Download"""

db_user_Email_DL.head(10)

"""### Data volume for Email UL (Bytes)"""

db_user_Email_UL = db.groupby(["MSISDN/Number"]).agg(Email_UL_Bytes = ("Email UL (Bytes)", 'sum')).sort_values(by='Email_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Email_UL

"""### Top 10 users with largest Email Upload"""

db_user_Email_UL.head(10)

"""### Data volume for Gaming DL (Bytes)"""

db_user_Gaming_DL = db.groupby(["MSISDN/Number"]).agg(Gaming_DL_Bytes = ("Gaming DL (Bytes)", 'sum')).sort_values(by='Gaming_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works

# db_user_DL_Gaming = db.groupby(["MSISDN/Number"]).agg({'Gaming DL (Bytes)':'sum'}) #it works but not sorted with Bytes

db_user_Gaming_DL

"""### Top 10 users with largest Gaming Download"""

db_user_Gaming_DL.head(10)

"""### Data volume for Gaming UL (Bytes)"""

db_user_Gaming_UL = db.groupby(["MSISDN/Number"]).agg(Gaming_UL_Bytes = ("Gaming UL (Bytes)", 'sum')).sort_values(by='Gaming_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Gaming_UL

"""### Top 10 users with largest Gaming Upload"""

db_user_Gaming_UL.head(10)

"""### Data volume for Other DL"""

db_user_other_DL = db.groupby(["MSISDN/Number"]).agg(Other_DL_Bytes = ("Other DL (Bytes)", 'sum')).sort_values(by='Other_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_other_DL

"""### Top 10 users with largest Other Services Download"""

db_user_other_DL.head(10)

"""### Data Volume for Other UL"""

db_user_other_UL = db.groupby(["MSISDN/Number"]).agg(Other_UL_Bytes = ("Other UL (Bytes)", 'sum')).sort_values(by='Other_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_other_UL

"""### Top 10 users with largest Other Services Download"""

db_user_other_UL.head(10)

"""#Top 10 users with largest Total Download"""

db_user_Total_DL = db.groupby(["MSISDN/Number"]).agg(Total_DL_Bytes = ("Total DL (Bytes)", 'sum')).sort_values(by='Total_DL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Total_DL.head(10)

"""### Top 10 users with largest Total Uploads"""

db_user_Total_UL = db.groupby(["MSISDN/Number"]).agg(Total_UL_Bytes = ("Total UL (Bytes)", 'sum')).sort_values(by='Total_UL_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
db_user_Total_UL.head(10)

"""### Top 3 most used applications"""

social_total = db["Social Media DL (Bytes)"].sum() + db["Social Media UL (Bytes)"].sum()

google_total = db["Google DL (Bytes)"].sum() + db["Google UL (Bytes)"].sum()
email_total = db["Email DL (Bytes)"].sum() + db["Email UL (Bytes)"].sum()
youtube_total = db["Youtube DL (Bytes)"].sum() + db["Youtube UL (Bytes)"].sum()
netflix_total = db["Netflix DL (Bytes)"].sum() + db["Netflix UL (Bytes)"].sum()
gaming_total = db["Gaming DL (Bytes)"].sum() + db["Gaming UL (Bytes)"].sum()
other_total = db["Other DL (Bytes)"].sum() + db["Other UL (Bytes)"].sum()

dict_bytes = {"Total Social Media Bytes":social_total, "Total Google Bytes":google_total,"Total Email Bytes":email_total, 
              "Total YouTube Bytes":youtube_total,"Total Netflix Bytes":netflix_total, 
              "Total Gaming Bytes": gaming_total, "Total Other Services Bytes":other_total}

max_3_keys = sorted(dict_bytes, key=dict_bytes.get, reverse=True)
for key in max_3_keys:
  print(f"{key} : {dict_bytes[key]}\n")

"""### Top 3 most used applications"""

max_3_keys = sorted(dict_bytes, key=dict_bytes.get, reverse=True)[:3]
for key in max_3_keys:
  print(f"{key} : {dict_bytes[key]}\n")

"""## K-Means clustering for each metric

### Normalize each metrics
* Frequency
* Duration of session - already normalized
* Total DL - already normalized
* Total UL -already normalized

#### Normalized Total DL
"""
minmax_scaler = preprocessing.MinMaxScaler()
def scalling_numeric_values(df):
  col_values = []
  col_exclude = ['Bearer Id', 'IMSI','MSISDN/Number','IMEI']
  for column in df.columns:
    if df[column].dtype == 'float64' and (column not in col_exclude):
      col_values.append(list(df[column].values))
  col_values_scaled = minmax_scaler.fit_transform(col_values)
  db_scaled = pd.DataFrame(col_values_scaled)
  return df

def scalling_numeric_values_0_1(df):
  for column in df.columns:
    col_exclude = ['Bearer Id', 'IMSI','MSISDN/Number','IMEI']
    if df[column].dtype == 'float64' and (column not in col_exclude):
      df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(150001,1))
  return df

db_sklearn = fill_missing_values(db.copy())
scalling_numeric_values(db_sklearn)

db_sklearn['Total DL (Bytes)'].head()

"""#### Normalized Total UL"""

db_sklearn['Total UL (Bytes)'].head()

"""#### Normalized Duration"""

db_sklearn['Dur. (ms)'].head()

# min_max_scaler = preprocessing.MinMaxScaler()

# dur_norm = (db_sklearn['Dur. (ms)'].values).reshape(-1,1) #returns a numpy array, reshape the feature
# dur_norm_scaled = min_max_scaler.fit_transform(dur_norm)
# db_sklearn[["Dur. (ms)"]] = pd.DataFrame(dur_norm_scaled)
# db_sklearn['Dur. (ms)'].head()

"""### Clustering Functions"""

kmeans = cluster.KMeans(n_clusters = 3, init = "k-means++", max_iter = 300
                  , n_init = 10, random_state=0)
#Clustering function
def Cluster(df,cols, title:str, xlabel:str, ylabel:str):
  X = df[cols].values
  # According to the elbow method, the nuber of clusters is 3
  
  # apply fit_predict -  map which sample to which cluster
  # and return number of clusters as single vector y K-means
  y_kmeans = kmeans.fit_predict(X)

  # Visualize the clusters
  plt.scatter(X[y_kmeans==0, 0],X[y_kmeans==0,1], s=100,c='red', label= "Cluster 1")
  plt.scatter(X[y_kmeans==1, 0],X[y_kmeans==1,1], s=100,c='blue', label= "Cluster 2")
  plt.scatter(X[y_kmeans==2, 0],X[y_kmeans==2,1], s=100,c='green', label= "Cluster 3")

  # plot Centroids
  plt.scatter(kmeans.cluster_centers_[:,0],
              kmeans.cluster_centers_[:,1],s=300,c='yellow', label='Centroids')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

  print(f"\nCentroids:\n{kmeans.cluster_centers_}\n\n")
  #Min-Max
  print(f"Min-Max Values \n\nCluster 1:\n\n")
  print(f"Max Value(ms) {X[y_kmeans==0,1].max()}\nMin Value(ms): {X[y_kmeans==0,1].min()}\nMean Value(ms):{X[y_kmeans==0,1].mean()}\nTotal Value(ms): {X[y_kmeans==0,1].sum()}\n\n")

  print(f"Cluster 2:\n\n")
  print(f"Max Value(ms) {X[y_kmeans==1,1].max()}\nMin Value(ms): {X[y_kmeans==1,1].min()}\nMean Value(ms):{X[y_kmeans==1,1].mean()}\nTotal Value(ms): {X[y_kmeans==1,1].sum()}\n\n")

  print(f"Cluster 3:\n\n")
  print(f"Max Value(ms) {X[y_kmeans==2,1].max()}\nMin Value(ms): {X[y_kmeans==2,1].min()}\nMean Value(ms):{X[y_kmeans==2,1].mean()}\nTotal Value(ms): {X[y_kmeans==2,1].sum()}\n\n")
  
  print("Optimal n_clusters - Elbow Method\n\n")
  n_clusters_elbow(X)


# elbow method to select n_clusters
def n_clusters_elbow(X):
  # uses within cluster sum of squares, WCSS
  wcss = []
  #fit kmeans to data and compute wcss
  for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter=300,n_init = 10, random_state = 0)
    kmeans.fit(X) # fit kmeans to dataset
  # append inertia_ - sum of squre distance of samples to their closest centroid
    wcss.append(kmeans.inertia_) 

  #plot elbow graph
  plt.plot(range(1,11), wcss)
  plt.title("Elbow Graph")
  plt.xlabel("Number of clusters")
  plt.ylabel("WCSS")
  plt.show()

"""### Session Frequncy Clusters"""

Cluster(db_sklearn,['MSISDN/Number', 'Bearer Id'], "Users' Session Clusters",'MSISDN/Number','Bearer Id')

"""### Duration Clusters"""

Cluster(db_sklearn,['MSISDN/Number', 'Dur. (ms)'], "Clusters of Duration (ms)",'MSISDN/Number','Duration (ms)')

"""### Min-Max for Non-Normalized Data"""

Cluster(db,['MSISDN/Number', 'Dur. (ms)'], "Clusters of Duration (ms)",'MSISDN/Number','Duration (ms)')

"""### Total Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Bytes'], "Total Bytes Clusters",'MSISDN/Number',"Total Bytes")

"""### Total Social Media Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Social Media (Bytes)'], "Total Social Media Bytes Clusters",'MSISDN/Number',"Total Social Media (Bytes)")

"""### Total Google Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Google (Bytes)'], "Total Google Bytes Clusters",'MSISDN/Number',"Total Google (Bytes)")

"""### Total Email Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Email (Bytes)'], "Total Email Bytes Clusters",'MSISDN/Number',"Total Email (Bytes)")

"""### Total Youtube Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Youtube (Bytes)'], "Total Youtube Bytes Clusters",'MSISDN/Number',"Total Youtube (Bytes)")

"""### Total Netflix Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Netflix (Bytes)'], "Total Netflix Bytes Clusters",'MSISDN/Number',"Total Netflix (Bytes)")

"""### Total Gaming Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Netflix (Bytes)'], "Total Netflix Bytes Clusters",'MSISDN/Number',"Total Netflix (Bytes)")

"""### Total Other Services Bytes Clusters"""

Cluster(db_totals_col,['MSISDN/Number', 'Total Other (Bytes)'], "Total Other Services Bytes Clusters",'MSISDN/Number',"Total Other (Bytes)")

"""### Total UL Clusters"""

Cluster(db_sklearn,['MSISDN/Number', 'Total UL (Bytes)'], "Clusters of Total UL",'MSISDN/Number',"Total UL (Bytes)")

"""### Min-Max for Non-Normalized Data"""

Cluster(db,['MSISDN/Number', 'Total UL (Bytes)'], "Clusters of Total UL",'MSISDN/Number',"Total UL (Bytes)")

"""### Data Scaling and Encoding

#### Drop some columns
"""

print("Numeric scaled data:\n")
db_sklearn.head() # scaled data

columns_drop = ['Start','Start ms', 'End', 'End ms', 'IMSI','IMEI','Last Location Name','Dur. (ms).1']
db_drop_col = db_sklearn.copy()
db_drop_col=db_sklearn.drop(columns_drop, axis=1)
print("Data with some columns dropped: \n")
db_drop_col.head()

db_drop_col.shape

db_encoded = db_drop_col.copy()
lb = LabelEncoder() 
column_encoded = ['Handset Manufacturer', 'Handset Type']


def db_encoding (df):
  for column in column_encoded:
    df[column] = lb.fit_transform(df[column])
  return df

db_encooded = db_encoding(db_encoded)
print("Data with categorical data encoded:\n")
db_encoded.head()

db_user_Duration.head()



def app():
    st.title("User Engagement Analysis")
    st.header('User Engagement Data')
    

    st.write("")
    st.header('Top 10 Numbers (Users) with highest')
    option = st.selectbox(
        'Top 10 Numbers (Users) with highest',
        ('Bearer Id', 'Number of Duration', 'Total Data Volume'))

    if option == 'Bearer Id':
        # data = clean_data_df.sort_values(
        #     'number of xDR Sessions', ascending=False).head(10)
        # name = 'number of xDR Sessions'
        db['Bearer Id'].value_counts().head(10)
    elif option == 'Number of Duration':
        # data = clean_data_df.sort_values('Dur (ms)', ascending=False).head(10)
        # name = 'Dur (ms)'
        db_user_Duration = db.groupby(["MSISDN/Number"]).agg(Duration_ms = ('Dur. (ms)', 'sum')).sort_values(by='Duration_ms', ascending = False)#.value_counts(ascending = False) # it also works
        db_user_Duration.head(10)
        
    elif option == 'Total Data Volume':
        # data = clean_data_df.sort_values(
        #     'Total Data Volume (Bytes)', ascending=False).head(10)
        # name = 'Total Data Volume (Bytes)'
        # db_Total_Bytes = db.copy()
        # db_Total_Bytes["Total Bytes"] = db["Total DL (Bytes)"] + db["Total UL (Bytes)"]
        db_user_total_bytes = db_Total_Bytes.groupby(["MSISDN/Number"]).agg(Total_Bytes = ("Total Bytes", 'sum')).sort_values(by='Total_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
        db_user_total_bytes.head(10)

    data = db_Total_Bytes.reset_index('MSISDN/Number')
    fig = px.pie(data, names='MSISDN/Number', values="Total Bytes")
    st.plotly_chart(fig)

    # st.write('You selected:', option)

    st.dataframe(data)

    st.write("")
    st.header('Top 10 Engaged Users Per App')
    app_option = st.selectbox(
        'Top 10 Engaged Users Per App',
        ('Social Media', 'Youtube','Google', 'Email', 'Netflix', 'Gaming', 'Other')
    )

    if app_option == 'Social Media':
        # app_data = app_clean_data_df.sort_values(
        #     'Social Media Data Volume (Bytes)',ascending=False
        # ).head(10)
        # db_Total_Bytes["Total Social Media (Bytes)"] = db["Social Media DL (Bytes)"] + db["Social Media UL (Bytes)"]
        # db_user_Social = db_totals_col.groupby(["MSISDN/Number"]).agg(Total_Social_Media_Bytes = ("Total Social Media (Bytes)", 'sum')).sort_values(by='Total_Social_Media_Bytes', ascending = False)#.value_counts(ascending = False) # it also works
        db_user_Social.head(10)
        
        app_name = 'Total Social Media (Bytes)'
    elif app_option == 'Youtube':
        # app_data = app_clean_data_df.sort_values(
        #     'Youtube Data Volume (Bytes)',ascending=False
        # ).head(10)
        db_user_Youtube.head(10)
        app_name = 'Total Youtube Bytes'
    elif app_option == 'Google':
        # app_data = app_clean_data_df.sort_values(
        #     'Google Data Volume (Bytes)',ascending=False
        # ).head(10)
        db_user_Google.head(10)
        app_name = 'Total Google Bytes'
    elif app_option == 'Email':
        # app_data = app_clean_data_df.sort_values(
        #     'Email Data Volume (Bytes)',ascending=False
        # ).head(10)
        db_user_Email.head(10)
        app_name = 'Total Email Bytes'
    elif app_option == 'Netflix':
        # app_data = app_clean_data_df.sort_values(
        #     'Netflix Data Volume (Bytes)',ascending=False
        # ).head(10)
        db_user_Netflix.head(10)
        app_name = 'Total Netflix Bytes'
    elif app_option == 'Gaming':
        # app_data = app_clean_data_df.sort_values(
        #     'Gaming Data Volume (Bytes)',ascending=False
        # ).head(10)
        db_user_Gaming.head(10)
        app_name = 'Total Gaming Bytes'
    else:
        # app_data = app_clean_data_df.sort_values(
        #     'Other Data Volume (Bytes)',ascending=False
        # ).head(10)
        db_user_Other.head(10)
        app_name = 'Total Other Services Bytes'
    
    app_data = app_data.reset_index('MSISDN/Number')
    app_fig = px.pie(app_data, names='MSISDN/Number', values=app_name)
    st.plotly_chart(app_fig)
    st.dataframe(app_data)

    st.title("User Clusters")
    st.write("")
    # eng_data_df = load_eng_data()
    st.dataframe(db_totals_col.head(1000))
    st.write("")
    st.markdown("***Users classified into 6 clusters based on their engagement(i.e. number of xDR sessions, duration and total data used).***")
    plotly_plot_scatter(db_totals_col, 'Total Bytes', 'Dur (ms)', 'cluster', 'xDR Sessions')


def plotly_plot_scatter(df, x_col, y_col, color, size):
    fig = px.scatter(df, x=x_col, y=y_col,
                 color=color, size=size)
    st.plotly_chart(fig)