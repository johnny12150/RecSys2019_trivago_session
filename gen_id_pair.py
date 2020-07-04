import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_seesion = pd.read_csv("data/train.csv", usecols=[0, 1, 2, 5])
# test_seesion = pd.read_csv("data/test.csv")
output_path = './data/baseline model/'

# 只保留 reference有值且是數字
train_seesion = train_seesion[train_seesion['reference'].apply(lambda x: str(x).isdigit())]
train_seesion['reference'] = train_seesion['reference'].astype(int)

# user id mapping成數字
le = LabelEncoder()
train_seesion['user_id'] = le.fit_transform(train_seesion['user_id'])

# TODO 只保留click out

# 留最後的5萬筆當validation
last_50k_sess = train_seesion.session_id.unique()[-50000:]
val = train_seesion[train_seesion.session_id.isin(last_50k_sess)]
train = train_seesion[~train_seesion.session_id.isin(last_50k_sess)]

train.drop(['session_id'], axis=1, inplace=True)  # drop col
val.drop(['session_id'], axis=1, inplace=True)
# 將資料整理成 user/ session_id 跟 item id pair的形式
# 先存成 txt (csv格式的)
train.to_csv(output_path + 'trivago_train_full.txt', sep='\t', index=False)
val.to_csv(output_path + 'trivago_val_full.txt', sep='\t', index=False)

