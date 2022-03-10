############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets/hitters.csv")
df.columns = [str(i).upper() for i in list(df.columns)]
df.head()

############################################
# EDA ANALIZI
############################################

check_df(df)

# BAĞIMLI DEĞİŞKEN ANALİZİ
df["SALARY"].describe()
sns.distplot(df["SALARY"])
plt.show()

sns.boxplot(df["SALARY"])
plt.show()

# KATEGORİK VE NUMERİK DEĞİŞKENLERİN SEÇİLMESİ
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# KATEGORİK DEĞİŞKEN ANALİZİ
rare_analyser(df, "SALARY", cat_cols)

# SAYISAL DEĞİŞKEN ANALİZİ
for col in num_cols:
    num_summary(df, col, plot=True)

# AYKIRI GÖZLEM ANALİZİ
for col in num_cols:
    print(col, check_outlier(df, col))

# Eksik Gözlemler kontrol ediliyor.
missing_values_table(df)

############################################
# FEATURE ENGINEERING
############################################

df['NEW_HIT_RATIO'] = df['HITS'] / df['ATBAT']
df['NEW_RUN_RATIO'] = df['HMRUN'] / df['RUNS']

df['NEW_CHIT_RATIO'] = df['CHITS'] / df['CATBAT']
df['NEW_CRUN_RATIO'] = df['CHMRUN'] / df['CRUNS']
df['NEW_AVG_ATBAT'] = df['CATBAT'] / df['YEARS']
df['NEW_AVG_HITS'] = df['CHITS'] / df['YEARS']
df['NEW_AVG_HMRUN'] = df['CHMRUN'] / df['YEARS']
df['NEW_AVG_RUNS'] = df['CRUNS'] / df['YEARS']
df['NEW_AVG_RBI'] = df['CRBI'] / df['YEARS']
df['NEW_AVG_WALKS'] = df['CWALKS'] / df['YEARS']

df["NEW_HIT_RAITO_YEAR"] = df['HITS'] / df['YEARS']
df["NEW_HMRUN_RAITO_YEAR"] = df['HMRUN'] / df['YEARS']
df["NEW_CHIT_RAITO_YEAR"] = df['ATBAT'] / df['YEARS']
df["NEW_RUN_RAITO_YEAR"] = df['RUNS'] / df['YEARS']
df["NEW_ASSIST_RAITO_YEAR"] = df['ASSISTS'] / df['YEARS']
df["NEW_ERROR_RAITO_YEAR"] = df['ERRORS'] / df['YEARS']
df["NEW_RBI_RAITO_YEAR"] = df['RBI'] / df['YEARS']
df["NEW_WALKS_RAITO_YEAR"] = df['WALKS'] / df['YEARS']

df["NEW_HIT_RAITO_YEAR_RATIO"] = df['NEW_HIT_RATIO'] / df["YEARS"]
df["NEW_RUN_RAITO_YEAR_RATIO"] = df['NEW_RUN_RATIO'] / df["YEARS"]

# One Hot Encoder
df = one_hot_encoder(df, cat_cols, drop_first=True)

############################################
# MODELLEME
############################################

df.dropna(inplace=True)

y = df['SALARY']
X = df.drop("SALARY", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)


##########################
# Tahmin Başarısını Değerlendirme
##########################


def control_y_pred(y_pred, y_test, y_train=None, y_train_pred=None, mse=True, rmse=False, mae=False):
    df = pd.DataFrame()
    if y_train is not None:
        df["Y"] = ["y_train", "y_test"]
        if mse:
            df["MSE"] = [mean_squared_error(y_train, y_train_pred),
                         mean_squared_error(y_test, y_pred)]
        if rmse:
            df["RMSE"] = [np.sqrt(mean_squared_error(y_train, y_train_pred)),
                          np.sqrt(mean_squared_error(y_test, y_pred))]
        if mae:
            df["MAE"] = [mean_absolute_error(y_train, y_train_pred),
                         mean_absolute_error(y_test, y_pred)]
    else:
        df["Y"] = ["y_test"]
        if mse:
            df["MSE"] = [mean_squared_error(y_test, y_pred)]
        if rmse:
            df["RMSE"] = [np.sqrt(mean_squared_error(y_test, y_pred))]
        if mae:
            df["MAE"] = [mean_absolute_error(y_test, y_pred)]

    print(df)


y_train_pred = reg_model.predict(X_train)
y_pred = reg_model.predict(X_test)

control_y_pred(y_pred=y_pred, y_test=y_test, y_train=y_train, y_train_pred=y_train_pred,
               mse=True, rmse=True, mae=True)
