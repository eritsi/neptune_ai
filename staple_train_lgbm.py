# +
import lightgbm as lgb
import neptune.new as neptune
from neptune.new.integrations.lightgbm import NeptuneCallback
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from neptune.new.types import File
import pandas_gbq as gbq

from lib.datasetLoader import datasetLoader
# -

dataset_loader = datasetLoader()
df = dataset_loader.load_by_file( "./test.sql" )

df.head()

df_train = df[df['client_year']!=2020]
df_test = df[df['client_year']==2020]

# Create run
run = neptune.init(
    project="eritsi/staple",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MTc5MWU2Ny1mZWJkLTRlYWYtYmY3Mi02MWY4YzdhMzg5ZDMifQ==",
    name="staple_train_trial",
    tags=["lgbm-integration", "train", "cls"]
)

# Create neptune callback
neptune_callback = NeptuneCallback(run=run)

# Define parameters
params = {
    'boosting_type': 'gbdt',
#     'num_leaves': 1024,
    'max_depth': -1,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample_for_bin': 200000,
    'objective': None,
    'class_weight': None,
    'min_split_gain': 0.0,
    'min_child_weight': 0.001,
#     'min_child_samples': 100,  # ParaTuning20,
    'subsample': 1.0,
    'subsample_freq': 0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': 0,
    'n_jobs': -1,
    'silent': True,
    'importance_type': 'split'
}

X_train, y_train = df_train[['product_code', 'comp_cluster',
                       'sales_count_before_1w', 'sales_count_before_2w', 'sales_count_before_3w', 'sales_count_before_4w', 
                       'mean_2w_sales_during_2m', 'std_sales_count', 'pred_yoy_8w_2wLAG' ]], df_train['sales_count']
X_test, y_test = df_test[['product_code', 'comp_cluster',
                       'sales_count_before_1w', 'sales_count_before_2w', 'sales_count_before_3w', 'sales_count_before_4w', 
                       'mean_2w_sales_during_2m', 'std_sales_count', 'pred_yoy_8w_2wLAG' ]], df_test['sales_count']
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, callbacks=[NeptuneCallback(run=run)],)

y_pred = model.predict(X_test)

# STEP4:
# Log evaluation results in tabular format
dataframe = pd.DataFrame(
    data={
        'y_test': y_test, 
        'y_pred': y_pred, 
        #'y_pred_probability':np.argmax(y_pred, axis=1),
    }
)
run['evaluation/predictions'].upload(File.as_html(dataframe))

# STEP 5:
# upload pickle file
run['model/pickled_model'].upload(File.as_pickle(model))



# Prepare data
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123
)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Train the model
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=["training", "validation"],
    callbacks=[neptune_callback],
)
