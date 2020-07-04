import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train_seesion = pd.read_csv("data/train.csv", usecols=[0, 1, 2, 5])
# test_seesion = pd.read_csv("data/test.csv")
output_path = './data/baseline model/'

# TODO 只保留click out


# 只保留 reference有值且是數字
train_seesion = train_seesion[train_seesion['reference'].apply(lambda x: str(x).isdigit())]
train_seesion['reference'] = train_seesion['reference'].astype(int)

# 過濾掉出現次數過少的 item
item_supports = train_seesion.groupby('reference').size()
train_seesion = train_seesion[np.in1d(train_seesion.reference, item_supports[item_supports>=5].index)]
session_lengths = train_seesion.groupby('session_id').size()
train_seesion = train_seesion[np.in1d(train_seesion.session_id, session_lengths[session_lengths>=2].index)]

# user id mapping成數字
# le = LabelEncoder()
# train_seesion['user_id'] = le.fit_transform(train_seesion['user_id'])
le = LabelEncoder()
train_seesion['session_id'] = le.fit_transform(train_seesion['session_id'])

train_seesion.rename(columns={'session_id': 'SessionId', 'timestamp': 'Time', 'reference': 'ItemId'}, inplace=True)
# 留最後的5萬筆當validation
last_50k_sess = train_seesion.SessionId.unique()[-50000:]
val = train_seesion[train_seesion.SessionId.isin(last_50k_sess)]
train = train_seesion[~train_seesion.SessionId.isin(last_50k_sess)]

train.drop(['user_id'], axis=1, inplace=True)  # drop col
val.drop(['user_id'], axis=1, inplace=True)
# 將資料整理成 user/ session_id 跟 item id pair的形式
# 先存成 txt (csv格式的)
train.to_csv(output_path + 'trivago_train_full.txt', sep='\t', index=False)
val.to_csv(output_path + 'trivago_val_full.txt', sep='\t', index=False)

