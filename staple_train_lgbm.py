import lightgbm as lgb
import neptune.new as neptune
from neptune.new.integrations.lightgbm import NeptuneCallback
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas_gbq as gbq
# from google.cloud import bigquery
# from google.cloud import bigquery_storage
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

# Create run
run = neptune.init(
    project="eritsi/staple",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MTc5MWU2Ny1mZWJkLTRlYWYtYmY3Mi02MWY4YzdhMzg5ZDMifQ==",
    name="staple_train_trial",
    tags=["lgbm-integration", "train", "cls"]
)

# Create neptune callback
neptune_callback = NeptuneCallback(run=run)

# Prepare GCP setting & tables
project = "eri-sandbox"
bqclient = bigquery.Client(project=project, location="asia-northeast1")
bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()

sql = '''
            SELECT
              *
            FROM
              `eri-sandbox.staple_result_dghbc_submission_sales_Mar.datamart_dghbc_avoid_leak`
            ORDER BY
              product_code
          '''

# read bq table through bqstorage_client
df = (
    bqclient.query(sql)
    .result()
    .to_dataframe(
        bqstorage_client=bqstorageclient
    )
)

#
df.head(5)

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

# Define parameters
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 10,
    "metric": ["multi_logloss", "multi_error"],
    "num_leaves": 21,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 12,
}

# Train the model
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=["training", "validation"],
    callbacks=[neptune_callback],
)