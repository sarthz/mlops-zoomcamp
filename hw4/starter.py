#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip freeze | grep scikit-learn')


# In[15]:


import pickle
import pandas as pd
import sys



taxi_type = sys.argv[1] #'yellow'
year = int(sys.argv[2]) #2022
month = int(sys.argv[3]) #2



# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[5]:

url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
print("URL is: ",url)
df = read_data(url)


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


print("Standard deviation of y_pred: ", y_pred.std())
print("Mean deviation of y_pred: ", y_pred.mean())


# In[11]:


df.head()


# In[20]:

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[21]:


output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


# In[24]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# In[25]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




