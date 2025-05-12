# Trip Duration Prediction

## Download Data

Make a directory data under notebooks

Download parquet files from NYC TLC Trip Record Data

* January 2021 Green Taxi Data
* February 2021 Green Taxi Data

```
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet
```

## Read Parquet Files

Read parquet files in pandas

```green_jan_2021 = pd.read_parquet('./data/green_tripdata_2021-01.parquet')```

Parquet reads in data as datetime

## Calculate duration of trip

Calculate duration of trip in minutes

```
green_jan_2021['duration'] = green_jan_2021.lpep_dropoff_datetime - green_jan_2021.lpep_pickup_datetime
green_jan_2021.duration = green_jan_2021.duration.apply(lambda td: td.total_seconds() / 60)
```

Visualize the duration

```
import seaborn as sns
import matplotlib.pyplot as plt
```

```sns.displot(green_jan_2021.duration)```


Check the distribution of trip duration

```green_jan_2021.duration.describe()```

Get the mean of trip duration

```((green_jan_2021.duration >= 1) & (green_jan_2021.duration <= 60)).mean()```

Limit trip durations to range of 1 to 60 minutes.

```green_jan_2021 = green_jan_2021[(green_jan_2021.duration >= 1) & (green_jan_2021.duration <= 60)]```

## Select features: categorical and numerical values

ID categorical and numerical values

```
categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']
```

Check types for categorical values

```green_jan_2021[categorical].dtypes```


Make categorical values string types

```green_jan_2021[categorical] = green_jan_2021[categorical].astype(str)```

Create dictionary of data from df

```from sklearn.feature_extraction import DictVectorizer```

```train_dicts = df[categorical + numerical].to_dict(orient='records')```

```
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
```


Get training set y values

```
target = 'duration'
y_train = df[target].values
```

Train the model with X_train and y_train

```from sklearn.linear_model import LinearRegression```

```
lr = LinearRegression()
lr.fit(X_train, y_train)
```

Apply linear regression model to X_train

```y_pred = lr.predict(X_train)```

Get prediction error to evaluate model performance

```
from sklearn.metrics import root_mean_squared_error
```

```
root_mean_squared_error(y_train, y_pred)
```

Visualize
```
sns.distplot(y_pred, label='prediction')
sns.distplot(y_train, label='actual')
plt.legend()
```

Create function that reads in the data, calculates trip duration, and limits it to range of 1 to 60 minutes

```
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df
```

Create training and validation sets

```
df_train = read_dataframe('./data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('./data/green_tripdata_2021-02.parquet')
```

Check length of data

```len(df_train), len(df_val)```

Combine categorical features

```
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
```

Select categorical and numerical features

```
categorical = ['PU_DO'] #['PULocationID, 'DOLocationID']
numerical = ['trip_distance]
```

Create data dictionaries for training and validation sets

```
dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
```

Create target values

```
target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values
```

Train linear regression model and evaluate the model

```
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

root_mean_squared_error(y_val, y_pred)
```

Train lasso model

```
from sklearn.linear_model import Lasso
```

```
lr = Lasso(0.01)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

root_mean_squared_error(y_val, y_pred)
```

Train ridge model

```
from sklearn.linear_model import Ridge
```

```lr = Ridge(0.01)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

root_mean_squared_error(y_val, y_pred)
```

## Save Model

```
import pickle
```

Create folder within notebooks directory
```
mkdir models
```

Save linear regression model

```
with open('models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)
```
