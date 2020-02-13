import pandas as pd

train_seesion = pd.read_csv("data/train.csv")
test_seesion = pd.read_csv("data/test.csv")
df_re = train_seesion.reset_index().set_index(['session_id', 'index'])
df_re = df_re[['user_id', 'session_id']].drop_duplicates()


def func(x):
    # print(isinstance(x, pd.DataFrame))  # to see whether if x is df
    return pd.DataFrame()


df = train_seesion[['user_id', 'session_id']].drop_duplicates()
# grouped = df.groupby(['user_id']).apply(func)  # will return a empty df
grouped = df.groupby(['user_id'])
print(grouped.describe())

# below has same result
# 1.
tables = df.pivot_table(columns='session_id', index='user_id', aggfunc='first')
print(tables.head())
# 2. this will trigger error way faster
df.set_index(['user_id', 'session_id']).unstack('session_id')  # Unstacked DataFrame is too big, causing int32 overflow
# 3.
tabs = pd.crosstab(index=df['user_id'], columns=df['session_id'])
print(tabs.head())

