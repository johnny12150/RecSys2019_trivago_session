#%%
import pandas as pd
from tqdm import tqdm
from metrics import mean_reciprocal_rank

train_seesion = pd.read_csv("data/train.csv")
test_seesion = pd.read_csv("data/test.csv")
# df_re = train_seesion.reset_index().set_index(['session_id', 'index'])

# 只保留 type是click的資料
train_click = train_seesion.loc[train_seesion.action_type == 'clickout item', ['user_id', 'session_id', 'reference', 'impressions']]
train_click['impressions'] = train_click['impressions'].apply(lambda x: x.split('|'))

#%%
# 觀察是不是 next-click都在 impression內
s_id = ''
first_step = True  # 判斷是不是session的開頭
previous_impressions = ''
previous_click = ''
pos_sum = 0
same_sum = 0 # 下一刻是不是點一樣
pred_count = 0
for i, data in tqdm(train_click.iterrows()):
    # 檢查有沒有換session
    if s_id == data['session_id']:
        first_step = False
        pred_count += 1
        # 看前一刻的 click是不是在現在的impression內
        if data['reference'] in previous_impressions:
            pos_sum += 1
        elif previous_click == data['reference']:
            same_sum += 1
    else:
        s_id = data['session_id']
        first_step = True
    previous_click = data['reference']
    previous_impressions = data['impressions']

# 算有多少比例是從impression點擊的
print(pos_sum/ pred_count)
print(same_sum/ pred_count)
print((pos_sum+same_sum)/ pred_count)

#%%
# 挑出全部 session倒數兩個 click
train_click['reference'] = train_click['reference'].astype(str)
train_click['refs'] = train_click.groupby(['session_id'])['reference'].apply(','.join)

# 計算 baseline MRR
# 1. random猜

# 2. 從 impression裡面 random挑

# 3. 直接猜前一刻的 click


# %%
