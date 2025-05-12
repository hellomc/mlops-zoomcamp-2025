# HW Trip Duration Prediction

## Download Data

Under directory notebooks/data/

Download parquet files from NYC TLC Trip Record Data

* January 2023 Yellow Taxi Data
* February 2023 Yellow Taxi Data

```
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet
```

## January Yellow Taxi Data

Read in January 2023 Yellow Taxi Data

```
yellow_jan_2023 = pd.read_parquet(./data/yellow_tripdata_2023-01.parquet)
```

Get number of columns and number of records

```
num_cols = yellow_jan_2023.shape[1]
num_records = yellow_jan_2023.shape[0]
print('Columns: ', num_cols)
print('Records: ', num_records)
```

Calculate duration of trip in minutes

```
yellow_jan_2023['duration'] = yellow_jan_2023.tpep_dropoff_datetime - yellow_jan_2023.tpep_pickup_datetime
yellow_jan_2023.duration = yellow_jan_2023.duration.apply(lambda td: td.total_seconds() / 60)
```

Check the distribution of trip duration to find the standard deviation

```
yellow_jan_2023.duration.describe()
```

Drop outliers of trip duration. Keep range 1 to 60 minutes

```
yellow_jan_2023 = yellow_jan_2023[(yellow_jan_2023.duration >= 1) & (yellow_jan_2023.duration <= 60)]
```

Calculate the percentage of records remaining after dropping the outliers.

```
new_num_records = yellow_jan_2023.shape[0]
print('Percentage Records Remaining: ', new_num_records / num_records)
```

## Select features

```categorical = ['PULocationID', 'DOLocationID']

yellow_jan_2023[categorical] = yellow_jan_2023[categorical].astype(str)
```

One Hot Encode

```
list_dicts = yellow_jan_2023[categorical].to_dict(orient='records')

dv = DictVectorizer()
feature_matrix = dv.fit_transform(list_dicts)

feature_matrix_arr = feature_matrix.toarray()

dimensionality = feature_matrix_arr.shape
print('Dimensionality:', dimensionality)
```

## Train the model

```
train_dicts = yellow_jan_2023[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
```

```
target = 'duration'
y_train = df[target].values
```

```
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)
```

Evaluate the model's performance against the training set
```
root_mean_squared_error(y_train, y_pred)
```

## Make function for reading in data

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

## Evaluate the model against Yellow Feb 2023 Taxi Data

Create training and validation sets

```
df_train = read_dataframe('./data/yellow_tripdata_2023-01.parquet')
df_val = read_dataframe('./data/yellow_tripdata_2023-02.parquet')
```

```
dv = DictVectorizer()

train_dicts = df_train[categorical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical].to_dict(orient='records')
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