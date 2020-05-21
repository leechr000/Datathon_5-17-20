import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


MARGIN = 7
# Set up ---------------------------------------------------------------------------------------------
path = "C:\\Users\\eelrs\\Documents\\Datathon_Spring\\bydate.csv"
data = pd.read_csv(path, parse_dates=['DATE'])

# one hot encode cat data -------------------------------------------------------------------------------
categorical_cols = ['NAME']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(data[categorical_cols]))
OH_cols.index = data.index

num_col = data.drop(categorical_cols, axis=1)
data = pd.concat([num_col, OH_cols], axis=1)

data = data.groupby(['DATE']).sum()

# feature generation -----------------------------------------------------------------------------------
col = data.columns
# recent profits (past 7 days)
month_b = data.TOTAL_PROFIT.rolling(MARGIN).sum()
month_b = month_b.rename('PAST_PROFIT')
month_b = month_b.to_frame()


# future profits (next 7 days)
month_for = data.TOTAL_PROFIT.shift(-MARGIN).rolling(MARGIN).sum()
month_for = month_for.rename('FUTURE_PROFIT')
month_for = month_for.to_frame()
feat = pd.concat([month_b, month_for], axis=1)
data = data.join(feat).fillna(value=0)

data = data.reindex(columns=data.columns)

data = data.iloc[MARGIN:-MARGIN]
print('--------------------------------------------------------------------------------------------')
print(data.describe())
print(data)
print('--------------------------------------------------------------------------------------------')

# feature selection for model -------------------------------------------------------------------------
y = data.FUTURE_PROFIT
X = data.drop(['FUTURE_PROFIT', 'Unnamed: 0', 'TOTAL_PROFIT'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)


# Model -------------------------------------------------------------------------------------------
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.5, random_state=0)

print('xgb_model....')
xgb_model.fit(X_train, y_train,
              early_stopping_rounds=5,
              eval_set=[(X_val, y_val)]
              )

sgd_model = SGDRegressor(learning_rate='optimal', random_state=0, early_stopping=True)
print('sgd_model....')
model = sgd_model.fit(X_train, y_train)
print(mean_squared_error(model.predict(X_val), y_val, squared=False))



