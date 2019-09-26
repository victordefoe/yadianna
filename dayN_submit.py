# encoding: utf-8
'''
@Author: 刘琛
@Time: 2019/9/26 23:00
@Contact: victordefoe88@gmail.com

@File: dayN_submit.py
@Statement:

'''

# coding: utf-8

# ### 这是第N天的尝试
#
# 验榜阶段-常规手段全套流程走一下
#
# 主要思路是用raw data做简单的去燥清洗以后，调用sklearn里面的xgboost + grid_serch调参 使用去预测
#
# 做一些特征的评估
#
# 特征处理：mean std max counts 梭哈大法
#

# 训练阶段
#
# 对train数据和A数据做同样的处理

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set(style="darkgrid")

# In[2]:





t_bs = pd.read_csv('data/1/train/train_bankStatement.csv')
t_bh = pd.read_csv('data/1/train/train_behaviors.csv')
t_cb = pd.read_csv('data/1/train/train_creditBill.csv')
t_pf = pd.read_csv('data/1/train/train_profile.csv')

train_sizes = {'bs': t_bs.shape[0], 'bh': t_bh.shape[0], 'cb': t_cb.shape[0], 'pf': t_pf.shape[0]}
trainset_ids = t_pf['用户标识'].values

print(trainset_ids)

A_bs = pd.read_csv('data/1/B/test_bankStatement_B.csv')
A_bh = pd.read_csv('data/1/B/test_behaviors_B.csv')
A_cb = pd.read_csv('data/1/B/test_creditBill_B.csv')
A_pf = pd.read_csv('data/1/B/test_profile_B.csv')

testA_ids = A_pf['用户标识'].values
print(testA_ids)

A_bs = pd.concat([t_bs, A_bs], axis=0)
A_bh = pd.concat([t_bh, A_bh], axis=0)
A_cb = pd.concat([t_cb, A_cb], axis=0)
A_pf = pd.concat([t_pf, A_pf], axis=0)

# ### 数据初步分析和预处理

# In[3]:


c = A_pf.shape[0]
a = len(A_bh['用户标识'].unique())
b = len(A_bs['用户标识'].unique())
d = len(A_cb['用户标识'].unique())
tol = pd.Series([a + b, c])
# A_pf.plot.pie(['教育程度'])
tol.plot.pie()
# plt.show()
print(tol)

# behaviour这个表的人的数量相比于正常人是不算少的

# 看看哪些人是比较既没有bh 又没有 bs的
a = A_bh['用户标识'].unique()
b = A_cb['用户标识'].unique()
c = A_pf['用户标识'].unique()  # 一共72490位客户
d = A_pf.loc[np.isin(c, a, invert=True), :]  # 这些是pf里面所有客户中，没有behaviour记录的客户 - 10804
e = A_pf.loc[np.isin(c, b, invert=True), :]  # 这些是pf里面所有客户中，没有credit card(信用卡)记录的客户 - 5465

keep_no_cb = e  ## 保存变量以后用
keep_no_bh = d

np.isin(c, a, invert=True)

# behaviour - credit card
nono = d.loc[np.isin(d['用户标识'], e['用户标识'], invert=False), :]  # 没有behaviour记录的客户中，也没有credit card记录的客户 - 783
# 总客户：72490
# e.loc[np.isin(e['用户标识'],d['用户标识'], invert=False),:]        #没有crdit card记录的客户中，也没有behaviour记录的客户 - 783
noyes = d.loc[np.isin(d['用户标识'], e['用户标识'], invert=True), :]  # 没有behaviour记录的客户中，有credit card记录的客户 - 10021
yesno = e.loc[np.isin(e['用户标识'], d['用户标识'], invert=True), :]  # 没有credit card记录的客户中，有behaviour记录的客户 - 4682

# 有credit card记录的客户中，又有behaviour记录的客户 - 57004
yesyes = A_pf.loc[
         np.isin(A_pf.loc[np.isin(c, a, invert=False), :]['用户标识'], A_pf.loc[np.isin(c, b, invert=False), :]['用户标识'],
                 invert=False), :]

bh_cb = pd.DataFrame(columns=['Behaviour-CreditCard'])

# bh_cb = pd.concat( [bh_cb, pd.Series(np.array([nono.shape[0], noyes.shape[0], yesno.shape[0], yesyes.shape[0]]))],axis=1, ignore_index=True)

bh_cb = bh_cb.reindex(['nono', 'noyes', 'yesno', 'yesyes'])
bh_cb.loc[:, 'Behaviour-CreditCard'] = np.array(
    [nono.shape[0], noyes.shape[0], yesno.shape[0], yesyes.shape[0]]).transpose()

bh_cb.plot.pie(['Behaviour-CreditCard'])

## ========================================================================


a = A_bh['用户标识'].unique()
b = A_bs['用户标识'].unique()
c = A_pf['用户标识'].unique()  # 一共72490位客户
d = A_pf.loc[np.isin(c, a, invert=True), :]  # 这些是pf里面所有客户中，没有behaviour记录的客户 - 10804
e = A_pf.loc[np.isin(c, b, invert=True), :]  # 这些是pf里面所有客户中，没有BankStatement记录的客户 - 5465

keep_no_bs = e  ## 保存变量以后用
np.isin(c, a, invert=True)

# Behaviour-BankStatement
nono = d.loc[np.isin(d['用户标识'], e['用户标识'], invert=False), :]  # 没有behaviour记录的客户中，也没有BankStatement记录的客户 - 783
# 总客户：72490
# e.loc[np.isin(e['用户标识'],d['用户标识'], invert=False),:]        #没有BankStatement记录的客户中，也没有behaviour记录的客户 - 783
noyes = d.loc[np.isin(d['用户标识'], e['用户标识'], invert=True), :]  # 没有behaviour记录的客户中，有BankStatement记录的客户 - 10021
yesno = e.loc[np.isin(e['用户标识'], d['用户标识'], invert=True), :]  # 没有BankStatement记录的客户中，有behaviour记录的客户 - 4682

# 有BankStatement记录的客户中，又有behaviour记录的客户 - 57004
yesyes = A_pf.loc[
         np.isin(A_pf.loc[np.isin(c, a, invert=False), :]['用户标识'], A_pf.loc[np.isin(c, b, invert=False), :]['用户标识'],
                 invert=False), :]

bh_bs = pd.DataFrame(columns=['Behaviour-BankStatement'])

# bh_cb = pd.concat( [bh_cb, pd.Series(np.array([nono.shape[0], noyes.shape[0], yesno.shape[0], yesyes.shape[0]]))],axis=1, ignore_index=True)

bh_bs = bh_bs.reindex(['nono', 'noyes', 'yesno', 'yesyes'])
bh_bs.loc[:, 'Behaviour-BankStatement'] = np.array(
    [nono.shape[0], noyes.shape[0], yesno.shape[0], yesyes.shape[0]]).transpose()

bh_bs.plot.pie(['Behaviour-BankStatement'])
bh_bs

# ==============================================================================


a = A_cb['用户标识'].unique()
b = A_bs['用户标识'].unique()
c = A_pf['用户标识'].unique()  # 一共72490位客户
d = A_pf.loc[np.isin(c, a, invert=True), :]  # 这些是pf里面所有客户中，没有credit card记录的客户
e = A_pf.loc[np.isin(c, b, invert=True), :]  # 这些是pf里面所有客户中，没有BankStatement记录的客户
np.isin(c, a, invert=True)

# Behaviour-BankStatement
nono = d.loc[np.isin(d['用户标识'], e['用户标识'], invert=False), :]  # 没有credit card记录的客户中，也没有BankStatement记录的客户
# 总客户：72490
# e.loc[np.isin(e['用户标识'],d['用户标识'], invert=False),:]        #没有BankStatement记录的客户中，也没有credit card记录的客户
noyes = d.loc[np.isin(d['用户标识'], e['用户标识'], invert=True), :]  # 没有credit card记录的客户中，有BankStatement记录的客户
yesno = e.loc[np.isin(e['用户标识'], d['用户标识'], invert=True), :]  # 没有BankStatement记录的客户中，有credit card记录的客户

# 有BankStatement记录的客户中，又有credit card记录的客户
yesyes = A_pf.loc[
         np.isin(A_pf.loc[np.isin(c, a, invert=False), :]['用户标识'], A_pf.loc[np.isin(c, b, invert=False), :]['用户标识'],
                 invert=False), :]

cb_bs = pd.DataFrame(columns=['CreditCard-BankStatement'])

# bh_cb = pd.concat( [bh_cb, pd.Series(np.array([nono.shape[0], noyes.shape[0], yesno.shape[0], yesyes.shape[0]]))],axis=1, ignore_index=True)

cb_bs = cb_bs.reindex(['nono', 'noyes', 'yesno', 'yesyes'])
cb_bs.loc[:, 'CreditCard-BankStatement'] = np.array(
    [nono.shape[0], noyes.shape[0], yesno.shape[0], yesyes.shape[0]]).transpose()

cb_bs.plot.pie(['CreditCard-BankStatement'])
cb_bs

compare = pd.concat([bh_cb, bh_bs, cb_bs], axis=1)

## ==========================================================================================
## 三无人员和三有人员
a = A_cb['用户标识'].unique()
b = A_bs['用户标识'].unique()
c = A_bh['用户标识'].unique()
d = A_pf['用户标识'].unique()  # 一共72490位客户

f1 = A_pf.loc[np.isin(c, a, invert=True), :]  # 没有credit card客户名单
f2 = f1.loc[np.isin(f1['用户标识'], b, invert=True), :]  # 也没有bankstatement 名单
f3 = f2.loc[np.isin(f2['用户标识'], c, invert=True), :]  # 也没有behaviour名单

g1 = A_pf.loc[np.isin(c, a, invert=False), :]  # 有credit card客户名单
g2 = g1.loc[np.isin(g1['用户标识'], b, invert=False), :]  # 也有bankstatement 名单
g3 = g2.loc[np.isin(g2['用户标识'], c, invert=False), :]  # 也有behaviour名单

# 总结一：
#
# 1. 显然这份数据，具有creditcard的数据，大部分人都有，占比是92.46%的人有信用卡。  85.09%的人有行为操作记录。  但是只有27.13%的人有对账单
#
# 2. 三无人员占比 %0.84 (611), 三有人员占比 10.39% (7535)
#
# 3.

# In[4]:


cc = compare['Behaviour-BankStatement']['yesyes'] + compare['Behaviour-CreditCard']['noyes']
cc / A_pf.shape[0]

g3.shape[0] / A_pf.shape[0]
# A_bs


# In[5]:


A = pd.merge(A_pf, A_bh, on='用户标识', how='left', sort=True)
# A.describe()
A.head()

# In[6]:


look_lb = pd.read_csv('data/1/train/train_label.csv')
look_lb['标签'].value_counts()
# 可以看到 5:1的比例，正负样本是稍微有些不平衡的


# ### 特征工程

# #### ** 处理A_pf **

# In[7]:


## 重新读入原始数据，不用每次改动了都重新run整个文件一次

t_pf = pd.read_csv('data/1/train/train_profile.csv')
A_pf = pd.read_csv('data/1/B/test_profile_B.csv')
A_pf = pd.concat([t_pf, A_pf], axis=0)

# In[8]:


import warnings

warnings.filterwarnings('ignore')

g3['full_info'] = 1.0
A_pf = pd.merge(A_pf, g3[['用户标识', 'full_info']], on='用户标识', how='left')
A_pf = A_pf.fillna(0.0)
A_pf

f3['less_info'] = 1.0
A_pf = pd.merge(A_pf, f3[['用户标识', 'less_info']], on='用户标识', how='left')
A_pf = A_pf.fillna(0.0)
A_pf.head()

keep_no_cb['no credit_card'] = 1.0
A_pf = pd.merge(A_pf, keep_no_cb[['用户标识', 'no credit_card']], on='用户标识', how='left')
A_pf = A_pf.fillna(0.0)
A_pf.head()

keep_no_bh['no behaviours'] = 1.0
A_pf = pd.merge(A_pf, keep_no_bh[['用户标识', 'no behaviours']], on='用户标识', how='left')
A_pf = A_pf.fillna(0.0)
A_pf.head()

keep_no_bs['no bankstatement'] = 1.0
A_pf = pd.merge(A_pf, keep_no_bs[['用户标识', 'no bankstatement']], on='用户标识', how='left')
A_pf = A_pf.fillna(0.0)
A_pf.drop_duplicates(['用户标识'], inplace=True)

# In[9]:


print(keep_no_bs.shape, keep_no_cb.shape, keep_no_bh.shape)
keep_no_cb.head()
# g3.head()
A_pf.head()

# In[10]:


A_pf.rename(columns={'户口类型': 'regis', '用户标识': 'id', '性别': 'gender', '职业': 'job', '教育程度': 'edu', '婚姻状态': 'marriage'},
            inplace=True)

# In[11]:


# A_pf = pd.read_csv('data/1/A/test_profile_A.csv')
A_pf.head()
b = pd.get_dummies(A_pf['gender'], prefix='gender').astype('float')
A_pf = pd.concat([A_pf, b], axis=1)
del A_pf['gender']
A_pf.head()

b = pd.get_dummies(A_pf['job'], prefix='job').astype('float')
A_pf = pd.concat([A_pf, b], axis=1)
del A_pf['job']
A_pf.head()

b = pd.get_dummies(A_pf['marriage'], prefix='marriage').astype('float')
A_pf = pd.concat([A_pf, b], axis=1)
del A_pf['marriage']
A_pf.head()

# In[12]:


t_lb = pd.read_csv('data/1/train/train_label.csv')
t_lb.shape
t_lb.rename(columns={'用户标识': 'id', '标签': 'label'}, inplace=True)
t_edus = A_pf[:train_sizes['pf']][['id', 'edu']]
t_edus['labels'] = t_lb['label']

t_edus.pivot(columns='edu', values='labels').sum().plot(kind='bar')
# 学历是3和4的 违约概率最高, 学历是1的基本不会违约

# # pd.concat(t_edus
# t_edus.describe()
# sns.distplot(t_edus['edu'], kde=True)
# # sns.plt.show()
# t_edus.hist('edu')
# sns.pairplot(t_edus[['labels','edu']])
# plt.show()

b = pd.get_dummies(A_pf['edu'], prefix='edu').astype('float')
A_pf = pd.concat([A_pf, b], axis=1)
del A_pf['edu']
A_pf.head()

# In[13]:


t_reg = A_pf[:train_sizes['pf']][['id', 'regis']]
t_reg['labels'] = t_lb['label']

t_reg.pivot(columns='regis', values='labels').sum().plot(kind='bar')

b = pd.get_dummies(A_pf['regis'], prefix='regis').astype('float')
A_pf = pd.concat([A_pf, b], axis=1)
del A_pf['regis']
A_pf.head()

# In[14]:


t_info = A_pf[:train_sizes['pf']][
    ['id', 'full_info', 'less_info', 'no credit_card', 'no behaviours', 'no bankstatement']]
t_info['label'] = t_lb['label']

t_info.pivot(columns='full_info', values='label').sum().plot(kind='bar')
plt.show()
t_info.pivot(columns='less_info', values='label').sum().plot(kind='bar')
plt.show()
t_info.pivot(columns='no credit_card', values='label').sum().plot(kind='bar')
plt.show()
t_info.pivot(columns='no behaviours', values='label').sum().plot(kind='bar')
plt.show()
t_info.pivot(columns='no bankstatement', values='label').sum().plot(kind='bar')
plt.show()

# In[15]:


print(pd.Series(t_info[t_info['no credit_card'].values == 1]['label'] == 1).sum() / pd.Series(
    t_info[t_info['no credit_card'].values == 1]['label'] == 1).count())
print(pd.Series(t_info[t_info['no bankstatement'].values == 1]['label'] == 1).sum() / pd.Series(
    t_info[t_info['no bankstatement'].values == 1]['label'] == 1).count())
print(pd.Series(t_info[t_info['no behaviours'].values == 1]['label'] == 1).sum() / pd.Series(
    t_info[t_info['no behaviours'].values == 1]['label'] == 1).count())
print(pd.Series(t_info[t_info['full_info'].values == 1]['label'] == 1).sum() / pd.Series(
    t_info[t_info['full_info'].values == 1]['label'] == 1).count())
print(pd.Series(t_info[t_info['less_info'].values == 1]['label'] == 1).sum() / pd.Series(
    t_info[t_info['less_info'].values == 1]['label'] == 1).count())

# 有这些特征的人中，违约人群占比，最高的是信息全的，信息越多越有可能违约

# sns.barplot(x=t_info.index,y=t_info.values)


# 这些都不能说明什么，这些特征并不强
#

# 综合处理A_pf！！

# In[16]:


A_pf.head()
A_pf.shape
# F_Apf 是最终处理好的，可以与其他表融合的变量， index必须是reset过的
F_Apf = A_pf
F_Apf.head()

# #### 处理 A_cb（test_creditBill_A.csv）

# 信用卡账单假设我们简单造2个特征 叫做：mean(上期还款金额-上期账单) ，另一个叫 mean(信用卡额度-本期消费)，删去各种金额，只保留额度
#
# 删去时间戳
#
# 银行标识 一人一行,造两个量，一个叫最常去的bank（独热），另一个叫去的不同bank的数量
#
#

# 将A_cb所有数据归一化

# In[17]:



t_cb = pd.read_csv('data/1/train/train_creditBill.csv')
A_cb = pd.read_csv('data/1/B/test_creditBill_B.csv')
A_cb = pd.concat([t_cb, A_cb], axis=0)

A_cb.rename(columns={'用户标识': 'id', '银行标识': 'card_bank', '账单时间戳': 'bill_ts',
                     '上期账单金额': 'last_bill', '上期还款金额': 'last_repay', '本期账单余额': 'now_bill', '信用卡额度': 'card_limit',
                     '还款状态': 'repay_state'}, inplace=True)

A_cb

z_score_scaler = lambda x: (x - np.mean(x)) / (np.std(x))
max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
sign_keep_scaler = lambda x: np.sign(x) * (np.abs(x) / (np.max(x) - np.min(x) + 1))

billmax = A_cb[['last_bill', 'last_repay', 'now_bill', 'card_limit']].abs().max().max()
billmin = A_cb[['last_bill', 'last_repay', 'now_bill', 'card_limit']].abs().min().min()
sign_mm_scaler = lambda x: np.sign(x) * (np.abs(x) / (billmax - billmin))
A_cb[['last_bill', 'last_repay', 'now_bill', 'card_limit']] = A_cb[
    ['last_bill', 'last_repay', 'now_bill', 'card_limit']].apply(sign_mm_scaler)

# In[18]:


## 预处理

A_cb.drop(A_cb[A_cb.bill_ts == 0].index, inplace=True)
# A_cb.shape


# In[19]:


# 上述两个数字还不一样，说明，有些人的额度变化然后又变了回去，需要统计变化了几次？变化的频率也许意味着什么
cnc = pd.DataFrame(A_cb.groupby(['id', 'card_bank'])['card_limit'].value_counts())
cnc.columns = ['limit_jump']
cnc.reset_index(inplace=True)

cnc.drop_duplicates(['id', 'card_bank'], keep='first', inplace=True)  # 保留信用卡长期停留的额度作为该卡的额度
users = cnc.drop(['card_limit', 'card_bank', 'limit_jump'], axis=1)  # 单独列出用户标识的一列以便之后的整合

# 处理信用卡的银行标识-每位用户分别拥有哪几家银行的卡？
banks = pd.get_dummies(cnc['card_bank'], prefix='card_bank')
card_idf = pd.concat([users, banks], axis=1).groupby(['id']).sum().reset_index()

# 处理信用卡额度-每位用户每张卡的额度有多少？
card_nc = pd.concat([users, banks], axis=1).drop(['id'], axis=1).multiply(cnc['card_limit'],
                                                                          axis=0).rename(
    columns=lambda x: 'bank-limit' + x[-2:])

card_nc = pd.concat([users, card_nc], axis=1).groupby(['id']).sum().reset_index()

card_info = pd.merge(card_idf, card_nc, on='id', how='left')  # 整合进总表card_info

# 处理还款情况-每位用户每张卡是否有欠款
repay = A_cb.groupby(['id', 'card_bank'])['repay_state'].max().reset_index().drop('card_bank', axis=1).groupby(
    ['id']).sum().reset_index()
card_info = pd.merge(card_info, repay, on='id', how='left')
card_info[card_info['repay_state'].values != 0]

# 处理额度跳变次数-每位用户每张卡的额度变化次数是不是多【这里日后可以加上时间信息，可能是额度变化/时间 代表这种变化是多长时间发生的】
card_jump = pd.concat([users, banks], axis=1).drop(['id'], axis=1).multiply(cnc['limit_jump'],
                                                                            axis=0).rename(
    columns=lambda x: 'bank-jump' + x[-2:])
card_jump = pd.concat([users, card_jump], axis=1).groupby(['id']).sum().reset_index()
card_jump
card_info = pd.merge(card_info, card_jump, on='id', how='left')
card_info

# 处理额度比一开始办卡时候增减了多少-每位用户每张卡的额度是增加还是减少，相比于一开始办卡时？
dif = A_cb.groupby(['id', 'card_bank'])['card_limit'].max() - A_cb.groupby(['id', 'card_bank'])['card_limit'].min()
dif[dif.values > 0].count()  # 一共有这么多用户的信用卡额度发生了变化
change = A_cb.groupby(['id', 'card_bank'])['card_limit'].last() - A_cb.groupby(['id', 'card_bank'])[
    'card_limit'].first()
change[change.values != 0].describe()

card_chg = pd.DataFrame(change)
card_chg.columns = ['limit_chg']
card_chg = card_chg.reset_index()
cnc['limit_chg'] = card_chg['limit_chg'].values

card_chg = pd.concat([users, banks], axis=1).drop(['id'], axis=1).multiply(cnc['limit_chg'],
                                                                           axis=0).rename(
    columns=lambda x: 'bank-limit_chg' + x[-2:])
# 将每一个人不同的银行卡额度变化取和
card_chg = pd.concat([users, card_chg], axis=1).groupby(['id']).sum().reset_index()
card_chg
card_info = pd.merge(card_info, card_chg, on='id', how='left')
card_info.head()

# In[20]:


card_info.columns

# In[21]:


loggest_card = A_cb.groupby('id')['card_bank'].agg(lambda x: x.value_counts().index[0]).reset_index(drop=True)

# In[22]:


card_num = A_cb.groupby(['id', 'card_bank']).count().reset_index().groupby('id')['card_bank'].count().reset_index(
    drop=True)

# In[23]:


card_info['card_num'] = card_num
card_info['longgest_card'] = loggest_card
card_info.head()

# In[24]:


# a = A_cb.groupby(['id','card_bank']).size().unstack(fill_value=0).reset_index(drop=True)
# del a.index.name
# pd.DataFrame(a.values).reindex()..apply(lambda x : 1.0 if x > 0 else 0.0)


# 同样地，我们想知道这些特征对的最终结果的影响

# In[25]:


t_lb = pd.read_csv('data/1/train/train_label.csv')
t_lb.rename(columns={'标签': 'label', '用户标识': 'id'}, inplace=True)

card_info.head()
A_cb.head()
cb_lb = t_lb[np.isin(t_lb['id'], card_info['id'])]
t_card_info = card_info[:cb_lb.shape[0]]
t_card_info = pd.merge(t_card_info, cb_lb, on='id', how='left')
t_card_info.head()

t_card_info.columns

# In[26]:


# t_card_info['card_num'].hist()

# 拥有卡的数量和最终违约的关系，卡越少，违约的人越多，因为总体人多
t_card_info.pivot(columns='longgest_card', values='label').sum().plot(kind='bar')

# 应该去分析每一类里面的概率
t_card_info[t_card_info['longgest_card'].values == 5]['label'].hist()

# In[27]:


# 使用时间最长的卡是哪张
long_dum = pd.get_dummies(card_info['longgest_card'], prefix='longcard').reset_index(drop=True)
card_info = pd.concat([card_info, long_dum], axis=1)
A_cb.columns

# In[28]:


# 人生的第一张卡是哪张
first_card = A_cb.sort_values(['id', 'bill_ts', 'card_bank']).groupby(['id']).first()['card_bank'].reset_index()
card_info = pd.concat([card_info, pd.get_dummies(first_card['card_bank'], prefix='fst_card')], axis=1)

# 终于可以开始处理时间-账单 相关的信息了

# In[29]:


A_cb.head()

# In[30]:


import time

# -2200000000
bill_ts = A_cb['bill_ts'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x - 3000000000)))

# In[31]:


# pd.to_datetime(A_cb['bill_ts'].apply(lambda x:x/10))
print(bill_ts.max())
print(bill_ts.min())

# In[32]:


A_cb['bill_ts'] = pd.to_datetime(bill_ts)

# In[33]:


A_cb['bill_ts'].hist()

# In[34]:


A_cb = A_cb.sort_values(['id', 'card_bank', 'bill_ts'], ascending=True)
# .groupby(['id','card_bank']).head(20)


# In[35]:


A_cb.head(35)

# 针对每张卡做如下分析：
#
#
# 1. 一个是每张卡办理的时长，办理时间是否很久了？
# 2. 使用的频率，办理后使用的次数 / 办理时长
# 3. 信用卡时间记录的离散程度，方差 (暂时只算了时间差均值)
#
# (我们不知道办卡时间，只知道第一次使用的时间)

# In[36]:


card_info.columns

# In[37]:


card_data = A_cb.groupby(['id', 'card_bank'])['last_repay'].mean().reset_index().drop(['last_repay'], axis=1)

# In[38]:


## 办卡时长
card_period = A_cb.groupby(['id', 'card_bank']).tail(1)['bill_ts'].reset_index(drop=True) - \
              A_cb.groupby(['id', 'card_bank']).head(1)['bill_ts'].reset_index(drop=True)
card_data['card_period'] = card_period

## 分析时间的密集程度。是否经常使用信用卡
# use_freq =
use_freq = (A_cb.groupby(['id', 'card_bank'])['bill_ts'].count().values - 1) / card_data['card_period'].apply(
    lambda x: x.days / 30)
use_freq = use_freq.replace(np.nan, 0)
use_freq = use_freq.replace(np.inf, 0)
use_freq.shape
# -np.log(a)

card_data['card_use_freq'] = use_freq

# In[39]:


## 计算同一张信用卡 每一条记录和上一条记录的时间差
time_delta = A_cb['bill_ts'] - A_cb.groupby(['id', 'card_bank'])['bill_ts'].shift(1)
time_delta.fillna(pd.Timedelta(0), inplace=True)
time_d_d = time_delta.apply(lambda x: x.days)

# In[40]:


A_cb.head()

# In[41]:



# # 距离上一次消费的时间
A_cb['last_time_gap'] = time_delta
# A_cb[time_d_d.abs()>100].head()
# # A_cb.groupby(['id','card_bank']).head()
# # time_delta
A_cb['gap_int'] = A_cb['last_time_gap'].dt.total_seconds()
avg_gap = pd.to_timedelta(A_cb.groupby(['id', 'card_bank'])['gap_int'].mean().round(), unit='s').reset_index(drop=True)

card_data['avg_gap'] = avg_gap
card_data
# ['last_time_gap']


# 综合处理A_cb！！

# In[42]:


## 处理 card_data
# 所有时间差转化为方便比较的int类型-天数
timedelta_cols = ['card_period', 'avg_gap']

card_data[timedelta_cols] = card_data[timedelta_cols].apply(lambda x: x.dt.days)

# In[43]:


card_data[timedelta_cols] = card_data[timedelta_cols].apply(lambda x: np.log(x + 1))
# banks.shape


# In[44]:


banks.shape[0] == card_data.shape[0]

banks

card_period = banks.reset_index(drop=True).multiply(card_data['avg_gap'], axis=0).rename(
    columns=lambda x: 'card_priod' + x[-2:])
card_period = pd.concat([users.reset_index(drop=True), card_period], axis=1).groupby(['id']).sum().reset_index(
    drop=True)
card_info = pd.concat([card_info, card_period], axis=1)
###
card_use_freq = banks.reset_index(drop=True).multiply(card_data['card_use_freq'], axis=0).rename(
    columns=lambda x: 'card_use_freq' + x[-2:])

card_use_freq = pd.concat([users.reset_index(drop=True), card_use_freq], axis=1).groupby(['id']).sum().reset_index(
    drop=True)
card_info = pd.concat([card_info, card_use_freq], axis=1)

###
card_avg_gap = banks.reset_index(drop=True).multiply(card_data['avg_gap'], axis=0).rename(
    columns=lambda x: 'avg_gap' + x[-2:])
card_avg_gap = pd.concat([users.reset_index(drop=True), card_avg_gap], axis=1).groupby(['id']).sum().reset_index(
    drop=True)
card_info = pd.concat([card_info, card_avg_gap], axis=1)

card_info
#
# card_jump
# card_info = pd.merge(card_info, card_jump,on='id', how = 'left')
# card_info
# card_data['card_period']


# In[45]:


print(card_info.columns)

# In[46]:


# F_Acb 是最终处理好的，可以与其他表融合的变量， index必须是reset过的
F_Acb = card_info
F_Acb.head()

# 信用卡额度特征很重要 - 平均额度，最大额度

# In[47]:


cards_mean_limit = A_cb.groupby(['id', 'card_bank']).card_limit.mean().reset_index().groupby(
    ['id']).card_limit.mean().reset_index().rename(columns={'card_limit': 'cards_mean_limit'})
F_Acb = pd.merge(F_Acb, cards_mean_limit, on='id', how='left')

# In[48]:


cards_max_limit = A_cb.groupby(['id', 'card_bank']).card_limit.max().reset_index().groupby(
    ['id']).card_limit.max().reset_index().rename(columns={'card_limit': 'cards_max_limit'})
F_Acb = pd.merge(F_Acb, cards_max_limit, on='id', how='left').head()

# 于是，下面是对每个人每张卡的所有账单做时间序列分析
#
#
# ---

# In[49]:


T = A_cb[['id', 'card_bank', 'bill_ts', 'now_bill']].reset_index(drop=True)
T = T.sort_values(['bill_ts']).set_index('bill_ts').groupby(['id', 'bill_ts']).mean()
# .resample('Y').head().ffill()


# In[50]:


years = A_cb['bill_ts'].apply(lambda x: x.year)
months = A_cb['bill_ts'].apply(lambda x: x.month)

A_cb['year'] = years
A_cb['month'] = months

# In[51]:


now_bill_avg = A_cb.groupby(['id', 'card_bank'])['now_bill'].rolling(3, min_periods=1).mean()

A_cb['now_bill_3monthavg'] = now_bill_avg.values

# In[52]:


A_cb.loc[(A_cb['id'] == 2) & (A_cb['card_bank'] == 5)]

A_cb.loc[(A_cb['id'] == 2) & (A_cb['card_bank'] == 5)][['bill_ts', 'last_bill']].set_index('bill_ts').plot()
A_cb.loc[(A_cb['id'] == 2) & (A_cb['card_bank'] == 5)][['bill_ts', 'now_bill_3monthavg']].set_index('bill_ts').plot()

#
#
# #### 处理A_bs （bankStatement）
#
#
#
# 盲猜4种情况：
# 工资标记为1的是对公转账,也就是公司发工资的情况
#
# is_salary 是 1 的情况下，trade_type全部是0，也就是说，trade_type 0 是指收入 1 是指支出
#
# is_salary 是 0 的情况，那就是转入钱或者消费，总体来看消费比转账收入要多
#
# 预先构建的特征有：工资收入总数，总收入和总支出

# In[53]:


A_bs = pd.read_csv('data/1/B/test_bankStatement_B.csv')
t_bs = pd.read_csv('data/1/train/train_bankStatement.csv')
A_bs = pd.concat([t_bs, A_bs], axis=0)

A_bs.rename(columns={'用户标识': 'id', '流水时间': 'bill_time', '交易类型': 'trade_type',
                     '交易金额': 'trade_amount', '工资收入标记': 'is_salary'}, inplace=True)

A_bs.head()

A_bs.head()

A_bs['real_bt'] = pd.to_datetime(A_bs.bill_time - 2300000000, unit='s')
A_bs['bt_onday'] = A_bs['real_bt'].dt.floor('d')
print('max-date:', A_bs.real_bt.max())
print('min-date:', A_bs.real_bt.min())
## 去除重复项
A_bs = A_bs.sort_values(['id', 'real_bt'])
# A_bs[A_bs.duplicated(keep = False)]
A_bs.drop_duplicates(inplace=True)

# In[54]:


A_bs.shape

# In[55]:


F_Abs = pd.DataFrame(columns=['id'])
F_Abs['id'] = A_bs['id'].unique()
F_Abs.head()

# In[56]:


A_bs.query('is_salary==1').trade_type.value_counts()

# In[57]:


A_bs['salary'] = A_bs['is_salary'] * A_bs['trade_amount']

A_bs['outcome'] = A_bs['trade_type'] * A_bs['trade_amount']
# 收入（包括工资也算）
A_bs['income'] = (1 - A_bs['trade_type']) * A_bs['trade_amount']

# In[58]:


F_Abs = pd.merge(F_Abs, A_bs.groupby('id')[['income', 'outcome']].sum().reset_index(), on='id', how='left')

F_Abs.rename(columns={'income': 'total_income', 'outcome': 'total_outcome'}, inplace=True)

# In[59]:


F_Abs = pd.merge(F_Abs, A_bs.groupby('id').salary.sum().reset_index(), on='id', how='left')

# In[60]:


A_bs.query('bill_time==0 ').trade_type.value_counts()

# 预处理把 时间为0的丢掉
A_bs.drop(A_bs[A_bs['bill_time'] == 0].index, inplace=True)

# 针对流水时间,只有去判断行为之间的频繁度，操作是否频繁，需要处理时间的period
#
# 尝试kmeans聚类离散化流水时间
#
#

# 1. 有记录的时间间隔是多久
# 2. 一天之内花出最多钱是多少
# 3. 每个人的日均工资
# 4. 工资发放的总次数
# 5. 工资发放次数/总的记录时间(按月份计算）（稳不稳定）
# 6. 工资发放平均间隔,没有的平均间隔的就置-1
# 7. 消费的总量/时间 （日均消费量）
# 8. 进项的总量/时间 （日均进项量）
# 9. 超过100块钱的消费天数（高消费天数）占比

# In[61]:


time_diff = (A_bs.groupby(['id']).bt_onday.last() - A_bs.groupby(['id']).bt_onday.first()).reset_index()
time_diff.columns = ['id', 'billtime_period']
time_diff.billtime_period += pd.to_timedelta('1 days')
time_diff.billtime_period = time_diff.billtime_period.apply(lambda x: x.days)
F_Abs = pd.merge(F_Abs, time_diff, on='id', how='left')

# In[62]:


day_cost_max = A_bs.query('trade_type==1').groupby(['id', 'bt_onday']).outcome.sum().reset_index().groupby(
    'id').outcome.max().reset_index().rename(columns={'outcome': 'day_cost_max'})
F_Abs = pd.merge(F_Abs, day_cost_max, on='id', how='left').fillna(0.0)  ## 有些人没有支出项，会有NAN，因而填充0

# In[63]:


# F_Abs.drop(columns=['billtime_period_x','day_cost_max_x','billtime_period_y', 'day_cost_max_y'], inplace=True)
F_Abs['day_salary'] = F_Abs.salary / F_Abs.billtime_period

# In[64]:


F_Abs = pd.merge(F_Abs, A_bs.groupby('id').is_salary.sum().reset_index().rename(columns={'is_salary': 'sal_count'}),
                 on='id', how='left')

# In[65]:


F_Abs['avg_sal_freq'] = F_Abs.sal_count / (F_Abs.billtime_period / 30)

# In[66]:


sal_items = A_bs.query('is_salary == 1')[['id']]
sal_items = pd.concat([sal_items, A_bs.query('is_salary == 1').groupby('id').bt_onday.apply(lambda x: x - x.shift(1))],
                      axis=1).dropna()

sal_items.bt_onday = sal_items.bt_onday.apply(lambda x: x.days)
sal_items = sal_items.groupby('id').mean().reset_index()
sal_items.columns = ['id', 'avg_sal_interval']
F_Abs = pd.merge(F_Abs, sal_items, on='id', how='left')
F_Abs.fillna(-1, inplace=True)

# In[67]:


F_Abs['day_avg_outcome'] = F_Abs.total_outcome / F_Abs.billtime_period

# In[68]:


F_Abs['day_avg_income'] = F_Abs.total_income / F_Abs.billtime_period

# In[69]:


F_Abs.day_avg_outcome.mean()

# 人均日销是203块钱，那么我们假定日销超过1000的就算高消费了,消费天数占比

# In[70]:


high_cost_days = A_bs.groupby(['id', 'bt_onday']).outcome.sum().reset_index().query(
    'outcome>=1000').groupby('id').count().reset_index().drop(columns=['bt_onday']).rename(
    columns={'outcome': 'high_cost_days'})

F_Abs = pd.merge(F_Abs, high_cost_days, on='id', how='left').fillna(0.0)

F_Abs.high_cost_days = F_Abs.high_cost_days / F_Abs.billtime_period
# .bill_time.reset_index().rename(columns={'bill_time':'high_cost'})


# In[71]:


F_Abs.head()

# In[72]:


A_bs.head()

# -----------------
#
# ----------------

# In[73]:


day_amount_mean = A_bs.query('trade_type==0').groupby(['bt_onday']).trade_amount.sum().sort_index()
day_amount_mean.plot()
day_amount_mean.describe()
day_amount_mean.sort_values().head()

# 交易类型1的金额大于交易类型0，但是交易类型1的金额数量抖动比较大，交易类型0就很平均，平均170刀

# In[74]:


day_people = A_bs.query('trade_type==1').groupby(['bt_onday']).id.count().sort_index()
day_people.plot()
day_people.sort_values().tail(20).sort_index()

# In[75]:


A_bs.query('id==1').sort_values('real_bt').query('salary==0').groupby('bt_onday').trade_amount.sum().plot()
A_bs.query('id==1 & salary == 0').head()

# In[76]:



### 聚类，，可能没必要

# idx = A_bs['bill_time']
# idx = pd.DataFrame(idx)
# max_idx = idx['bill_time'].max()

# k = 10 # 聚类数
# from sklearn.cluster import KMeans #引入KMeans
# kmodel = KMeans(n_clusters = k, n_jobs = 4) #建立模型，n_jobs是并行数，一般等于CPU数较好
# kmodel.fit(idx) #训练模型
# c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0) #输出聚类中心，并且排序（默认是随机序的）
# w = pd.rolling_mean(c, 2).iloc[1:] #相邻两项求中点，作为边界点
# w = [0] + list(w[0]) + [max_idx] #把首末边界点加上


# In[77]:


# d3 = pd.cut(idx['bill_time'], w, labels = range(k))
# A_bs['bill_time'] = d3
# A_bs.head()
# del A_bs['流水时间']
# A_bs.head()


# 盲猜工资标记是指的是否是工资收入，显然 工资1 * 金额 就是 这个人的收入状况！

# In[78]:


F_Abs.head()

# #### 处理A_bh
#
# 这个很难弄，弄不好会有反效果，时间关系暂时不做它
#
# 59种子类型2,111种子类型1： 尝试用xgboost选取最合适的top60个特征、
#
# 时间处理：针对时间序列，首先要知道，如何将时间离散化。将日期分类为几个类型。星期几可以离散化，同时我认为星期几与行为类型有强关联，这在后面特征工程处可以造特征
#

# **01 把数据分成A t test三个表，分别t test用来分析数据，A 用来处理数据**

# In[273]:


## 读入数据
t_bh = pd.read_csv('data/1/train/train_behaviors.csv')
test_bh = pd.read_csv('data/1/B/test_behaviors_B.csv')
A_bh = pd.concat([t_bh, test_bh], axis=0)

# In[274]:


t_bh.rename(columns={'用户标识': 'id', '日期': 'date', '星期几': 'day',
                     '行为类型': 'bh_type', '子类型1': 'sub_bh1', '子类型2': 'sub_bh2'}, inplace=True)

# 分析日期和最终标签的关系
# errors='coerce', 异常值将被设置为NaT
t_date = pd.to_datetime(t_bh['date'], format='%m-%d', errors='coerce')
t_bh['date'] = t_date
t_bh.dropna(axis=0, how='any', inplace=True)

# In[275]:


test_bh.rename(columns={'用户标识': 'id', '日期': 'date', '星期几': 'day',
                        '行为类型': 'bh_type', '子类型1': 'sub_bh1', '子类型2': 'sub_bh2'}, inplace=True)

# 分析日期和最终标签的关系
# errors='coerce', 异常值将被设置为NaT
test_date = pd.to_datetime(test_bh['date'], format='%m-%d', errors='coerce')
test_bh['date'] = test_date
test_bh.dropna(axis=0, how='any', inplace=True)

# In[276]:


A_bh.rename(columns={'用户标识': 'id', '日期': 'date', '星期几': 'day',
                     '行为类型': 'bh_type', '子类型1': 'sub_bh1', '子类型2': 'sub_bh2'}, inplace=True)

# 分析日期和最终标签的关系
# errors='coerce', 异常值将被设置为NaT
date = pd.to_datetime(A_bh['date'], format='%m-%d', errors='coerce')
A_bh['date'] = date
A_bh.dropna(axis=0, how='any', inplace=True)

# In[277]:


A_bh.head()

# In[278]:


A_bh = A_bh.sort_values(['id', 'date'])
t_bh = t_bh.sort_values(['id', 'date'])
test_bh = test_bh.sort_values(['id', 'date'])

# In[279]:


F_Abh = pd.DataFrame()
F_Abh['id'] = A_bh.id.unique()

# **02 分析处理date这个变量**

# In[280]:


test_bh['date'].value_counts().sort_index().plot()

# In[281]:


t_bh['date'].value_counts().sort_index().plot()

# 根据上图可以看出训练集和测试集总体分布是一致的，操作最频繁的时间是11月份左右，再看看违约发生的用户的操作的发生时间

# In[282]:


t_lb = pd.read_csv('data/1/train/train_label.csv')
t_lb.shape
t_lb.rename(columns={'用户标识': 'id', '标签': 'label'}, inplace=True)

# In[283]:


t_bh = pd.merge(t_bh, t_lb, on='id', how='left')

# In[284]:


t_bh[t_bh.label == 1]['date'].value_counts().sort_index().plot()
# 违约的人和不违约人操作次数比
t_bh['label'].value_counts()

# In[285]:


# 平均操作次数
peoples = \
t_bh.groupby(['label', 'id'])['date'].count().reset_index().rename(columns={'date': 'ops_num'}).groupby(['label'])[
    'id'].count().reset_index().rename(columns={'id': 'num_people'})
opss = t_bh['label'].value_counts().reset_index()
opss.columns = ['label', 'num_ops']
(opss['num_ops'] / peoples['num_people'])

# 差别不大，差不多,试一试根据操作的次数频率，把一年的日期按照频次分成不同的tag

# In[286]:


bins = (A_bh['date'].value_counts().sort_index() // 30000)
A_bh['date'].value_counts().sort_index().plot()

# In[287]:


# A_bh['date_tag'] = A_bh['date'].apply(lambda x : bins[x])
A_bh['date_tag'] = bins[A_bh['date']].values

# In[288]:


A_bh['date_tag'].value_counts()
datetag_people = A_bh.groupby(['date_tag', 'id'])['date'].count().reset_index().groupby('date_tag')[
    'id'].count().reset_index().rename(columns={'id': 'num_people'})
# 在这些时间标签里面 每个时段的 操作的用户数
datetag_people.plot(x='date_tag')

# A_bh.groupby('id').count()
# .groupby(['date_tag'])['id'].count()


# In[289]:


date_people = A_bh.groupby(['date', 'id'])['day'].count().reset_index().groupby('date')[
    'id'].count().reset_index().rename(columns={'id': 'num_people'})
date_people.plot(x='date')

# 可以分析得出 基本上操作的次数和操作的人数在“数量-日期”上的分布是接近的。更多人在这里操作，而不是这个时段的人们有更多操作

# **分析日期和标签的关系，可以 在哪些日期（几月份）操作比较多的人会不喜欢还款，做一个可视化**
#

# In[290]:


# 对于每个日期来说，违约用户曲线：也就是说，违约用户更喜欢在哪个时间段去操作
bad_date = t_bh.groupby(['date', 'id'])['label'].mean().reset_index().groupby(
    ['date', 'label']).count().reset_index().rename(columns={'id': 'num_people'}).groupby(
    ['label', 'date']).mean().reset_index()
bad_date[bad_date.label == 0].plot(x='date', y='num_people')
bad_date[bad_date.label == 1].plot(x='date', y='num_people')

# In[291]:


t_bh_u = t_bh.groupby('id')['label'].mean().reset_index()
t_bh_u['bh_period'] = (
t_bh.sort_values(['id', 'date']).groupby('id')['date'].last() - t_bh.sort_values(['id', 'date']).groupby('id')[
    'date'].first()).reset_index(drop=True)
t_bh_u

# In[292]:


t_bh_u.groupby(['label', 'bh_period']).count().reset_index().rename(columns={'id': 'num_user'}).query('label==1').plot(
    x='bh_period', y='num_user')
tbhu_by_day = t_bh_u.groupby(['label', 'bh_period']).count().reset_index().rename(columns={'id': 'num_user'})
tbhu_by_day['bh_period_rate'] = tbhu_by_day.groupby('label')['num_user'].apply(lambda x: x / x.sum())

# 大致趋势也是相同的。。

# **03 分析 day 这个变量**

# In[293]:


t_bh.groupby(['day', 'id'])['label'].count().reset_index().groupby('day')['id'].count().plot(x='day', y='num_people',
                                                                                             kind='bar')

# In[294]:


# 查看 人数-星期几 分布比例
day_label = t_bh.groupby(['day', 'id'])['label'].mean().reset_index()
day_label_num = day_label.groupby(['label', 'day']).count().reset_index().rename(columns={'id': 'num_people'})
day_label_num['rate'] = day_label_num.groupby('label').apply(
    lambda x: x['num_people'] / x['num_people'].sum()).reset_index(drop=True)
day_label_num.plot(x='day', y='num_people', color='green', kind='bar')
day_label_num[day_label_num.label == 0].rate.values - day_label_num[day_label_num.label == 1].rate.values

# day_label_num[day_label_num].plot(x = 'day', y='num_people',color='green')


# 1. 人们更不喜欢在周四操作
# 2. 分布很平均
#
# 创造特征如下：
# 1. 最常星期几去操作
# 2. 固定只在星期几操作（分析出现的星期几的个数）

# In[295]:


freq_max_days = A_bh.groupby(['id', 'day'])['date'].count().reset_index().sort_values(['id', 'date']).groupby(
    'id').last().reset_index().rename(
    columns={'day': 'freq_max_days'}).drop(columns=['date'])
freq_max_days

F_Abh = pd.merge(F_Abh, freq_max_days, on='id', how='left')

F_Abh = pd.concat([F_Abh, pd.get_dummies(F_Abh.freq_max_days, prefix='freq_max_days')], axis=1)

del F_Abh['freq_max_days']

# In[296]:


days_appear = A_bh.groupby(['id', 'day'])['date'].count().reset_index().groupby('id').day.count().reset_index().rename(
    columns={'day': 'days_appear'})
F_Abh = pd.merge(F_Abh, days_appear, on='id', how='left')

# In[297]:


F_Abh.head()

# **04 总体分析：处理记录条数和分析时间间隔**

# In[298]:


A_bh_t = A_bh.sort_values(['id', 'date'])

# In[299]:


time_diff = (A_bh_t.groupby('id')['date'].last() - A_bh_t.groupby('id')['date'].first()).reset_index()
time_diff.columns = ['id', 'bh_period']
time_diff.head()

# In[300]:



F_Abh = pd.merge(F_Abh, time_diff, on='id', how='left')
F_Abh.head()

# 时间间隔和最后标签的关系

# In[301]:


zero_day_u = t_bh_u[t_bh_u.bh_period == '0 days'].id
zero_day_bh = t_bh[np.isin(t_bh.id, zero_day_u)]
# zer_day_bh = t_bh


# In[302]:


# 只有一天记录的人们，他们这一天的记录的操作的次数
# 人数 随着 操作次数不同 的变化情况
zero_day_bh.groupby(['label', 'id'])['date'].count().reset_index().rename(
    columns={'date': 'num_bh'}).query('label==0').num_bh.value_counts().reset_index().rename(
    columns={'index': 'num_bh', 'num_bh': 'count'}).sort_values('num_bh').plot(x='num_bh', y='count')

zero_day_bh.groupby(['label', 'id'])['date'].count().reset_index().rename(
    columns={'date': 'num_bh'}).query('label==1').num_bh.value_counts().reset_index().rename(
    columns={'index': 'num_bh', 'num_bh': 'count'}).sort_values('num_bh')[:200].plot(x='num_bh', y='count')
# ['num_bh'].value_counts().reset_index().sort_values('index')

## 这一段是写的意思是，统计每一个操作行为次数下，有多少人是有这么多条记录的；大概能看出来的是，只有一天记录的人中，前200次操作记录以内是人数最集中的，一般人集中在60次到120次
## 因为大部分人只有一天的记录，一天的操作一般是60-120次


# 可以看出，大部分人的行为都是0，也就是说只有一天的操作记录
#
# 有些人是真的奇怪，他们是怎么达到上两万条记录的

# In[303]:


A_bh.groupby('id')['date'].count().reset_index().query('id==39')
## 日均操作次数
a = A_bh.groupby('id')['date'].count().reset_index().sort_values('id')
b = pd.concat([F_Abh.id, F_Abh.bh_period.apply(lambda x: x.days) + 1], axis=1).sort_values('id').reset_index(drop=True)
c = pd.concat([a.id, a['date'] / b['bh_period']], axis=1).rename(columns={0: 'count_perday'})
F_Abh = pd.merge(F_Abh, c, on='id', how='left')

## 操作最频繁一天操作的次数
F_Abh = pd.merge(F_Abh, A_bh.groupby(['id', 'date'])['day'].count().reset_index().rename(
    columns={'day': 'num_ops'}).reset_index().groupby('id')['num_ops'].max().reset_index().rename(
    columns={'num_ops': 'freq_max'}), on='id', how='left')

# 每位用户操作大于120次的天数
opsbigger120 = A_bh.groupby(['id', 'date'])['day'].count().reset_index().rename(
    columns={'day': 'num_ops'}).reset_index().query('num_ops>120').groupby('id').num_ops.count().reset_index().rename(
    columns={'num_ops': 'ops>120'})

F_Abh = pd.merge(F_Abh, opsbigger120, on='id', how='left')
F_Abh.fillna(0.0, inplace=True)

# **05 处理行为类型相关**

# In[304]:


## 选择操作最多和最少的bh_type作为特征好了
max_bh_type = A_bh.groupby('id').bh_type.value_counts().rename(columns={'bh_type': 0}).reset_index().groupby(
    'id').first().bh_type.reset_index().rename(columns={'bh_type': 'max_bh_type'})
F_Abh = pd.merge(F_Abh, max_bh_type, on='id', how='left')
min_bh_type = A_bh.groupby('id').bh_type.value_counts().rename(columns={'bh_type': 0}).reset_index().groupby(
    'id').last().bh_type.reset_index().rename(columns={'bh_type': 'min_bh_type'})
F_Abh = pd.merge(F_Abh, min_bh_type, on='id', how='left')

b = pd.get_dummies(F_Abh['max_bh_type'], prefix='max_bh_type').astype('float')
F_Abh = pd.concat([F_Abh, b], axis=1)
del F_Abh['max_bh_type']
b = pd.get_dummies(F_Abh['min_bh_type'], prefix='min_bh_type').astype('float')
F_Abh = pd.concat([F_Abh, b], axis=1)

del F_Abh['min_bh_type']

# 处理bh_type
#
# 用Xgboost选择哪个子类型最重要

# In[305]:


# F_Abh.drop(columns=['max_ops_day_0', 'max_ops_day_1', 'max_ops_day_2', 'max_ops_day_3','max_ops_day_4', 'max_ops_day_5', 'max_ops_day_6', 'max_ops_day_7'], inplace=True)
# F_Abh.drop(columns=['max_bh_type_x','max_bh_type_y'], inplace=True)


# In[306]:


F_Abh.columns

# In[307]:


A_bh.query('bh_type==1').sub_bh1.value_counts().count()

# In[308]:


F_Abh.bh_period = F_Abh.bh_period.apply(lambda x: x.days)
## 以月份为单位计算时长
F_Abh['bh_period_mon'] = F_Abh.bh_period / 30

# 1. 子类型1的总次数除以天数（每种子类型1的月均操作频次）
# 2. 子类型2的总次数除以天数（每种子类型1的月均操作频次）
# 3. 最常使用的子类型1
# 4. 最常使用的子类型2

# In[309]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

from xgboost import plot_importance
from matplotlib import pyplot

# In[310]:


subbh_sel = pd.concat([A_bh.id, pd.get_dummies(A_bh.sub_bh1, prefix='sub1_fq')], axis=1)

## 姑且认为那些只有一天记录的人，一个月内也每天重复这个操作？
months = F_Abh.bh_period_mon
months.replace(0, 1 / 30, inplace=True)

subbh_1_freq = subbh_sel.groupby(['id']).sum().reset_index()
subbh_1_freq[subbh_1_freq.columns[1:]] = subbh_1_freq[subbh_1_freq.columns[1:]].apply(lambda x: x / (months))

t_lb = pd.read_csv('data/1/train/train_label.csv')
t_lb.rename(columns={'用户标识': 'id', '标签': 'label'}, inplace=True)
bh_labels = pd.merge(subbh_1_freq, t_lb, on='id', how='left').dropna().label
sub_bh1_freq_set = pd.merge(subbh_1_freq, t_lb, on='id', how='left').dropna().drop(columns=['id', 'label'])

import warnings

warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(subbh_1_freq_set, bh_labels, test_size=0.3, random_state=0)

select = XGBClassifier(cv=5, error_score='raise',
                       estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                               colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                                               max_depth=6, min_child_weight=1, missing=None, n_estimator=50,
                                               n_estimators=100, n_jobs=6, nthread=None,
                                               objective='binary:logistic', random_state=0, reg_alpha=0,
                                               reg_lambda=1, scale_pos_weight=1, seed=None, silent=False,
                                               subsample=1))
select.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', verbose=1)

# In[311]:


# 特征选择
features_list = pd.Series(X_train.columns.values)
importances = pd.Series(select.feature_importances_)
cs = pd.concat([features_list, importances], axis=1).sort_values(1, ascending=False)[:20][0]
sub_bh1_freq_sel = pd.concat([subbh_1_freq.id, subbh_1_freq[cs]], axis=1)
F_Abh = pd.merge(F_Abh, sub_bh1_freq_sel, on='id', how='left')

# In[312]:


subbh_sel2 = pd.concat([A_bh.id, pd.get_dummies(A_bh.sub_bh2, prefix='sub2_fq')], axis=1)

subbh_2_freq = subbh_sel2.groupby(['id']).sum().reset_index()

subbh_2_freq[subbh_2_freq.columns[1:]] = subbh_2_freq[subbh_2_freq.columns[1:]].apply(lambda x: x / (months))

# In[313]:


bh_labels = pd.merge(subbh_2_freq, t_lb, on='id', how='left').dropna().label
sub_bh2_freq_set = pd.merge(subbh_2_freq, t_lb, on='id', how='left').dropna().drop(columns=['id', 'label'])

import warnings

warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(subbh_2_freq_set, bh_labels, test_size=0.3, random_state=0)

select = XGBClassifier(cv=5, error_score='raise',
                       estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                               colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                                               max_depth=6, min_child_weight=1, missing=None, n_estimator=50,
                                               n_estimators=100, n_jobs=6, nthread=None,
                                               objective='binary:logistic', random_state=0, reg_alpha=0,
                                               reg_lambda=1, scale_pos_weight=1, seed=None, silent=False,
                                               subsample=1))
select.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', verbose=1)

# In[314]:


# 特征选择
features_list = pd.Series(X_train.columns.values)
importances = pd.Series(select.feature_importances_)
cs = pd.concat([features_list, importances], axis=1).sort_values(1, ascending=False)[:20][0]
sub_bh2_freq_sel = pd.concat([subbh_2_freq.id, subbh_2_freq[cs]], axis=1)
F_Abh = pd.merge(F_Abh, sub_bh2_freq_sel, on='id', how='left')

# In[315]:


## 最常使用子类型1
max_sub1 = A_bh.groupby(['id']).sub_bh1.value_counts().rename(columns={'sub_bh1': 0}).reset_index().groupby(
    ['id']).first().reset_index().drop(columns=[0])
max_sub2 = A_bh.groupby(['id']).sub_bh2.value_counts().rename(columns={'sub_bh2': 0}).reset_index().groupby(
    ['id']).first().reset_index().drop(columns=[0])

# In[316]:


max_sub1_dum = pd.concat([max_sub1, pd.get_dummies(max_sub1.sub_bh1, prefix='max_sub1')], axis=1).drop(
    columns=['sub_bh1'])
max_sub1_dum.head()
max_sub1_label = pd.merge(max_sub1_dum, t_lb, on='id', how='left').dropna().drop(columns=['id']).dropna().label
max_sub1_set = pd.merge(max_sub1_dum, t_lb, on='id', how='left').dropna().drop(columns=['id', 'label']).dropna()

import warnings

warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(max_sub1_set, max_sub1_label, test_size=0.3, random_state=0)

select = XGBClassifier(cv=5, error_score='raise',
                       estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                               colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                                               max_depth=6, min_child_weight=1, missing=None, n_estimator=50,
                                               n_estimators=100, n_jobs=6, nthread=None,
                                               objective='binary:logistic', random_state=0, reg_alpha=0,
                                               reg_lambda=1, scale_pos_weight=1, seed=None, silent=False,
                                               subsample=1))
select.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', verbose=1)

# In[323]:


features_list = pd.Series(X_train.columns.values)
importances = pd.Series(select.feature_importances_)
cs = pd.concat([features_list, importances], axis=1).sort_values(1, ascending=False)[:20][0]
max_sub1_set = pd.concat([max_sub1_dum.id, max_sub1_dum[cs]], axis=1)

F_Abh = pd.merge(F_Abh, max_sub1_set, on='id', how='left')

# In[324]:


max_sub2_dum = pd.concat([max_sub2, pd.get_dummies(max_sub2.sub_bh2, prefix='max_sub2')], axis=1).drop(
    columns=['sub_bh2'])
max_sub2_dum.head()
max_sub2_label = pd.merge(max_sub2_dum, t_lb, on='id', how='left').dropna().drop(columns=['id']).dropna().label
max_sub2_set = pd.merge(max_sub2_dum, t_lb, on='id', how='left').dropna().drop(columns=['id', 'label']).dropna()

import warnings

warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(max_sub2_set, max_sub2_label, test_size=0.3, random_state=0)

select = XGBClassifier(cv=5, error_score='raise',
                       estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                               colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                                               max_depth=6, min_child_weight=1, missing=None, n_estimator=50,
                                               n_estimators=100, n_jobs=6, nthread=None,
                                               objective='binary:logistic', random_state=0, reg_alpha=0,
                                               reg_lambda=1, scale_pos_weight=1, seed=None, silent=False,
                                               subsample=1))
select.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', verbose=1)

# In[325]:


features_list = pd.Series(X_train.columns.values)
importances = pd.Series(select.feature_importances_)
cs = pd.concat([features_list, importances], axis=1).sort_values(1, ascending=False)[:20][0]
max_sub2_sel = pd.concat([max_sub2_dum.id, max_sub2_dum[cs]], axis=1)
F_Abh = pd.merge(F_Abh, max_sub2_sel, on='id', how='left')

# In[ ]:


# fig,ax = plt.subplots(figsize=(10,10))
# plot_importance(select,
#                height=0.5,
#                 ax=ax,
#                max_num_features=64)
# plt.show()



# In[326]:


F_Abh = pd.merge(F_Abh, A_bh.groupby(['id']).date.count().reset_index().rename(columns={'date': 'bh_count'}), on='id',
                 how='left')
F_Abh['count_permon'] = F_Abh.bh_count / F_Abh.bh_period_mon

# **06 结合时间序列相关，处理行为类型 **
# 1. 分析这个人操作最频繁的行为类型，每一个类型和上一个相同类型操作发生的间隔
# > 1.1 bhtype - 每个人1天之内操作最多的次数
# >
# 1.2 bhtype - 每个人都需要补充
#

# In[327]:


F_Abh.columns.values

# In[328]:


daymax_bh_type = pd.concat([A_bh[['id', 'date']], pd.get_dummies(
    A_bh.bh_type, prefix='daymax_bh_type')], axis=1).groupby(['id', 'date']).sum().reset_index().groupby(
    ['id']).max().reset_index().drop(columns=['date'])
F_Abh = pd.merge(F_Abh, daymax_bh_type, on='id', how='left')

# In[329]:


F_Abh.head()

# **Behaviour - END 最终分析-特征选择 **

# In[132]:


bh_lb = pd.merge(t_lb.rename(columns={'用户标识': 'id', '标签': 'label'}), pd.DataFrame(F_Abh.id), on='id',
                 how='right').dropna()
bh_train = pd.merge(F_Abh, bh_lb, on='id', how='right')

# In[133]:


bh_corr = bh_train.corr()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(bh_corr, vmax=1, square=True)

#  ### 组合A_pf和A_cb,A_bs

# In[330]:


F_Abh.head()

# In[331]:


# F_Abh.bh_period = F_Abh.bh_period.apply(lambda x:x.days)


# In[332]:


## 归一化，效果好吗？未必
F_Abh_m = F_Abh
keep = ['bh_period', 'count_perday', 'freq_max', 'count_permon', 'bh_count']
F_Abh_m[keep] = F_Abh[keep].apply(max_min_scaler)

F_Apf_m = F_Apf
F_Acb_m = F_Acb
F_Abs_m = F_Abs

# In[333]:


## 对bs的金额做归一化

F_Abs_m[
    ['total_income', 'total_outcome', 'salary', 'day_cost_max', 'day_salary', 'day_avg_outcome', 'day_avg_income']] = \
F_Abs[['total_income', 'total_outcome', 'salary', 'day_cost_max', 'day_salary', 'day_avg_outcome',
       'day_avg_income']].apply(max_min_scaler)

# In[334]:


F_Abs_m.head()

# In[335]:


# 组合A_pf和A_cb

F_Acb_m.columns.values
sets = pd.merge(F_Apf_m, F_Acb_m, on='id', how='left')
# A_bs.head()
# A_bs[list(~A_bs.columns.duplicated(keep='first'))]
# A_bs.columns

# sets = pd.merge(sets, A_bs, on='用户标识', how='left')





# 查看缺失值,给没有信用卡的填0

# In[336]:


print(sets.shape[0] - A_cb.shape[0])
print(sets.shape)
print(A_bs.shape)

sets.columns
A_bs.columns

sets.isnull().sum(axis=0)
# sets[train_sizes['pf']:].isnull().sum()

## 填充0

sets = sets.fillna(0.0)
sets[sets.columns[-12:]].tail(60)
sets.head()

# A_cb.tail(60)



sets = pd.merge(sets, F_Abh_m, on='id', how='left')

sets.shape

## 有NAN的列
sets.isnull().sum()[sets.isnull().sum() > 0]

# In[337]:


sets.fillna(0, inplace=True)

# zzy的特征

# In[338]:


## zzy 特征
zzy = pd.read_csv('./featurezzy.csv')

zzy.drop(columns='Unnamed: 0', inplace=True)
keep = ['用户标识', '流水余额', '流水余额类型', '交易0频次', '交易1频次', '总额度']
rename_dict = {'用户标识': 'id', '流水余额': 'bs_balance', '流水余额类型': 'bs_balance_sign', '交易0频次': 'trade0_count',
               '交易1频次': 'trade1_count', '总额度': 'all_limit'}
zzy = zzy[keep]
zzy.rename(columns=rename_dict, inplace=True)

# In[339]:


zzy.head()

# In[340]:


F_Abs.columns

# In[341]:


sets = pd.merge(sets, zzy, on='id', how='left')

# 分训练和测试

# In[342]:


# seperate
train_set = sets[np.isin(sets.id, trainset_ids)]
A_set = sets[np.isin(sets.id, testA_ids)]
train_set.shape
A_set.shape

train_set.astype('float')
A_set.astype('float')
# A_set[A_set['工资0'].isnull().values == True].shape
A_set.head()

# > miss_idx = train_set[train_set['bank-limit_1'].isnull()].index
#
# > miss_idx.shape
#
# > train_set = train_set.drop(miss_idx, axis=0)
#
# >train_set = train_set.fillna(0.0)
#
# > \# 好在A-set里面的人信用卡信息都齐全，否则填充0会对结果有较大影响,这部分填充的0是bankstate的
# A_set = A_set.fillna(0.0)
#
# 之前试过的思考是： 这么多人可能并不在cb内，也就是说，他们并没有信用卡，一共是64436人，把这5465人除去，剩下的所有NAN用0填充
#
# 但是这样会造成不平衡，没有信用卡 这一个重要特征没用上。不妨填充0试试

# In[343]:


## 融合 F_Abs
train_set = pd.merge(train_set, F_Abs_m, on='id', how='left').fillna(-1)
A_set = pd.merge(A_set, F_Abs_m, on='id', how='left').fillna(-1)

# In[344]:


A_set.shape

# ### 调用模型训练和预测-模型调优
#
# 1是违约 0是不违约，预测它是1的概率

# In[348]:


t_lb = pd.read_csv('data/1/train/train_label.csv')
t_lb.rename(columns={'用户标识': 'id', '标签': 'label'}, inplace=True)
# t_lb = t_lb.drop(miss_idx, axis=0)
t_lb.shape

train_set.drop(['id'], axis=1, inplace=True)
train_set.head()

# 先什么也不改地训练一波，
#
# 1. 实际上代价函数是需要改进的，自定义mectric很重要
# 2. 调参
# 3. 可视化重要特征
#
# gridsize 搜索调参

# In[191]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

from xgboost import plot_importance
from matplotlib import pyplot

# ----------
# 为了最快速验证结果，选择不用grid_search做，但是为了后面，统一使用grid_search作为模型变量名
import warnings

warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(train_set, t_lb['label'], test_size=0.3, random_state=0)

grid_search = XGBClassifier(cv=5, error_score='raise',
                            estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                    colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                                                    max_depth=6, min_child_weight=1, missing=None, n_estimator=50,
                                                    n_estimators=100, n_jobs=6, nthread=6,
                                                    objective='binary:logistic', random_state=0, reg_alpha=0,
                                                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=False,
                                                    subsample=1))
grid_search.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', verbose=1)
# grid_search.fit(X_train,y_train)


# In[164]:


## import warnings
warnings.filterwarnings('ignore')

X_train, X_val, y_train, y_val = train_test_split(train_set, t_lb['label'], test_size=0.5, random_state=0)

clf = XGBClassifier(n_jobs=20, learning_rate=0.1, n_estimator=50, max_depth=6, silent=False,
                    objective='binary:logistic')
# clf.fit(X_train,y_train, eval_set=[(X_train, y_train), (X_val, y_val)],eval_metric='auc',verbose=1 )
param_test = {
    'n_estimators': [200, 250, 300, 350],
    'max_depth': [3, 4, 5, 6, 7, 8, 9]
}  # 大杀器XGBoost
grid_search = GridSearchCV(n_jobs=6, estimator=clf, param_grid=param_test, scoring='accuracy', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# In[192]:


test = A_set.drop(['id'], axis=1, inplace=False)

# fig,ax = plt.subplots(figsize=(20,20))
# plot_importance(clf,
#                height=0.5,
#                 ax=ax,
#                max_num_features=100)

probs = pd.Series(grid_search.predict_proba(test)[:, 1])
probs[(probs.values > 0.5) == True].shape

# In[ ]:


test_id = pd.Series(testA_ids, name='id')
upload = pd.concat([test_id, probs], axis=1)

# In[ ]:


# del upload['index']
upload
upload.to_csv('./out/upload.csv', header=0, index=0)

# In[404]:


from sklearn.preprocessing.data import OneHotEncoder

a = pd.DataFrame(np.array([[1, 2, 3], [1, 4, 6], [2, 2, 2]]))

xgbenc = OneHotEncoder()
b = xgbenc.fit_transform(a)
pd.DataFrame(b.toarray())

# In[432]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import numpy as np
import pdb
from sklearn.preprocessing.data import OneHotEncoder


class XgboostFeature():
    ##可以传入xgboost的参数
    ##常用传入特征的个数 即树的个数 默认30
    def __init__(self, n_estimators=30, learning_rate=0.3, max_depth=3, min_child_weight=1, gamma=0.3,
                 subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1,
                 reg_alpha=1e-05, reg_lambda=1, seed=27):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.nthread = nthread
        self.scale_pos_weight = scale_pos_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seed = seed
        print('Xgboost Feature start, new_feature number:', n_estimators)

    def mergeToOne(self, X, X2):
        #         X2 = pd.DataFrame(X2)
        #         ## 编码为onehot
        #         xgbenc = OneHotEncoder()
        #         X2 = pd.DataFrame(xgbenc.fit_transform(X2).toarray())

        #         ## 对于所有的列做onehot编码
        X3 = pd.concat([X, X2], axis=1)
        return X3
        ##切割训练

    def fit_model_split(self, X_train, y_train, X_test, y_test):
        ## 先用已有特征训练XGBoost模型，然后利用XGBoost模型学习到的树来构造新特征，最后把这些新特征加入原有特征一起训练模型。
        ## 构造的新特征向量是取值0/1的，向量的每个元素对应于XGBoost模型中树的叶子结点。
        ## 当一个样本点通过某棵树最终落在这棵树的一个叶子结点上，那么在新特征向量中这个叶子结点对应的元素值为1，而这棵树的其他叶子结点对应的元素值为0。
        ## 新特征向量的长度等于XGBoost模型里所有树包含的叶子结点数之和。最后将新的特征扔到LR模型进行训练。
        ## 实验结果表明xgboost+lr能取得比单独使用两个模型都好的效果

        ##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)
        clf = XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            nthread=self.nthread,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            seed=self.seed)
        clf.fit(X_train_1, y_train_1)
        y_pre = clf.predict(X_train_2)
        y_pro = clf.predict_proba(X_train_2)[:, 1]
        print("pred_leaf=T AUC Score : %f" % metrics.roc_auc_score(y_train_2, y_pro))
        print("pred_leaf=T  Accuracy : %.4g" % metrics.accuracy_score(y_train_2, y_pre))
        new_feature = clf.apply(X_train_2)
        X_train_new2 = self.mergeToOne(X_train_2, new_feature)
        new_feature_test = clf.apply(X_test)
        X_test_new = self.mergeToOne(X_test, new_feature_test)
        print("Training set of sample size 0.4 fewer than before")
        return X_train_new2, y_train_2, X_test_new, y_test
        ##整体训练

    def fit_model(self, X_train, y_train, X_test, y_test=None):
        clf = XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            nthread=self.nthread,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            seed=self.seed)
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        y_pro = clf.predict_proba(X_test)[:, 1]
        if y_test is not None:
            print("pred_leaf=T  AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro))
            print("pred_leaf=T  Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre))
        new_feature = pd.DataFrame(clf.apply(X_train)).reset_index(drop=True)
        nt_ids = new_feature.shape[0]
        new_feature_test = pd.DataFrame(clf.apply(X_test)).reset_index(drop=True)
        nf = pd.concat([new_feature, new_feature_test], axis=0)
        xgbenc = OneHotEncoder()
        nf_dum = pd.DataFrame(xgbenc.fit_transform(nf).toarray())
        #         pdb.set_trace()
        new_feature = nf_dum[:nt_ids].reset_index(drop=True)
        new_feature_test = nf_dum[nt_ids:].reset_index(drop=True)
        X_train_new = self.mergeToOne(X_train, new_feature)

        X_test_new = self.mergeToOne(X_test, new_feature_test)
        print("Training set sample number remains the same")
        return X_train_new, y_train, X_test_new, y_test


# In[433]:


model = XgboostFeature(n_estimators=50)
new_trainset, new_trainset_lb, new_test, _ = model.fit_model(train_set, t_lb['label'], test)

# In[435]:


new_test.shape

#
# ---------------------- 2019年9月14日23:48:56
#
# ### 可视化权重的重要性 + 处理策略

# In[169]:


### 丢掉Abs特征

# train_set.drop(columns = F_Abs_m.columns.drop('id'), inplace=True)
# A_set.drop(columns = F_Abs_m.columns.drop('id'), inplace=True)


# In[436]:




import matplotlib.pylab as plt

X_train, X_val, y_train, y_val = train_test_split(new_trainset, t_lb['label'], test_size=0.3, random_state=0)
analys = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                       colsample_bytree=1, gamma=0.1, learning_rate=0.1, max_delta_step=0,
                       max_depth=6, min_child_weight=1, missing=None, n_estimators=300,
                       n_jobs=6, nthread=6, objective='binary:logistic', random_state=0,
                       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                       silent=False, subsample=1)

analys.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc')
# analys.fit(X_train,y_train,eval_metric='auc')


# In[437]:


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig, ax = plt.subplots(figsize=(20, 20))
plot_importance(analys,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

# In[ ]:


features_list = pd.Series(X_train.columns.values)
importances = pd.Series(analys.feature_importances_)
pd.concat([features_list, importances], axis=1).sort_values(1, ascending=False)[:20]

# In[ ]:


train_set.columns.values

# In[439]:


probs = pd.Series(analys.predict_proba(new_test)[:, 1])
probs[(probs.values > 0.5) == True].shape

# In[ ]:


# clf.booster
fi = -analys.feature_importances_
cols = pd.DataFrame(X_train.columns)
cols_keep = X_train.columns[[cols.loc[fi.argsort()][:20].index]]

# rename_cols(train_set)
X_train, X_val, y_train, y_val = train_test_split(train_set[cols_keep], t_lb['label'], test_size=0.5, random_state=0)
X_train.shape

# In[ ]:


A_set = sets[train_sizes[3]:].drop(['用户标识'], axis=1)
rename_cols(A_set)
# cols_keep
A_set = A_set[cols_keep]
A_set.head()

# In[ ]:


X_train.shape

# In[ ]:


clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                    max_depth=4, min_child_weight=1, missing=None, n_estimators=100,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=False, subsample=1)
train_set.head()
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc', verbose=1)

# param_test = {
#     'n_estimators': [30,32,34,36,38,40,42,44,46,48,50],
#     'max_depth': [2,3,4,5,6,7]
# }#大杀器XGBoost
# grid_search = GridSearchCV(estimator=clf , param_grid=param_test , scoring='accuracy',cv=5, verbose=1)
# grid_search.fit(X_train,y_train)


# In[441]:


# probs = pd.Series(analys.predict_proba(A_set)[:,1])
probs[(probs.values > 0.5) == True].shape

# In[442]:


test_id = pd.Series(testA_ids, name='id')
upload = pd.concat([test_id, probs], axis=1)

upload
upload.to_csv('./out/upload.csv', header=0, index=0)

# In[ ]:


probs.head()

# In[ ]:


A_set.head()
A_set.columns

A_set[['超额还款金额均值', '省钱指数', '信用卡额度']]

# In[ ]:


test = A_set.drop(['用户标识'], axis=1, inplace=False)
test_id = A_set['用户标识']
test_id = test_id.reset_index()
# test_id = test_id['用户标识']
test_id.head()
# result = clf.predict_proba(test)


# In[ ]:


# print(grid_search.grid_scores_)
print(grid_search.best_params_)

grid_search.best_score_

# In[354]:


result = grid_search.predict_proba(test)
result[:100]

# 查看一下预测的违约的人有多少个，输出结果保存到csv里面

# In[355]:


probs = pd.Series(result[:, 1])
probs[(probs.values > 0.5) == True].shape
# probs.shape
# result.shape


# In[ ]:


upload = pd.concat([test_id, probs], axis=1)
del upload['index']
upload
upload.to_csv('./out/upload.csv', header=0, index=0)

# ### KS评估

# In[157]:


import scipy as sp
from scipy import stats


# In[158]:


def ks_calc_2samp(data, score_col, class_col):
    Bad = data.ix[data[class_col[0]] == 1, score_col[0]]
    Good = data.ix[data[class_col[0]] == 0, score_col[0]]
    data1 = Bad.values
    data2 = Good.values
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = (np.searchsorted(data2, data_all, side='right')) / (1.0 * n2)
    ks = np.max(np.absolute(cdf1 - cdf2))
    cdf1_df = pd.DataFrame(cdf1)
    cdf2_df = pd.DataFrame(cdf2)
    cdf_df = pd.concat([cdf1_df, cdf2_df], axis=1)
    cdf_df.columns = ['cdf_Bad', 'cdf_Good']
    cdf_df['gap'] = cdf_df['cdf_Bad'] - cdf_df['cdf_Good']
    return ks, cdf_df


# In[ ]:


probs

# In[440]:


# scores = pd.Series(grid_search.predict_proba(X_val)[:,1]).reset_index(drop=True)
scores = pd.Series(analys.predict_proba(X_val)[:, 1]).reset_index(drop=True)
classes = pd.Series(y_val).reset_index(drop=True)

data = pd.DataFrame()
data[0] = scores
data[1] = classes
data.columns = ['pred', 'label']

ks_2samp, cdf_2samp = ks_calc_2samp(data, ['pred'], ['label'])
ks_2samp

# In[ ]:


data.head()

# In[ ]:


k_value, p_value = sp.stats.ks_2samp(scores, classes)
k_value

# ### reference

# In[ ]:


from sklearn.model_selection import KFold, StratifiedKFold


def get_ctr(train, test, cols, C):
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = train[['用户标识'] + cols]
    oof['ctr_{}'.format('_'.join(cols))] = 0
    predictions = test[['用户标识'] + cols]
    feature_importance_df = pd.DataFrame()
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train)):
        X_train = train.iloc[trn_idx]
        X_valid = train.iloc[val_idx]
        t_ctr = X_train.groupby(cols)['标签'].apply(lambda x: sum(x) / (len(x) + C)).reset_index()
        t_ctr.columns = cols + ['ctr_{}'.format('_'.join(cols))]
        X_valid = pd.merge(X_valid, t_ctr, on=cols, how='left', copy=False)
        oof['ctr_{}'.format('_'.join(cols))][val_idx] = X_valid['ctr_{}'.format('_'.join(cols))].values
    t_ctr = train.groupby(cols)['标签'].apply(lambda x: sum(x) / (len(x) + C)).reset_index()
    t_ctr.columns = cols + ['ctr_{}'.format('_'.join(cols))]
    oof_test = pd.merge(predictions, t_ctr, on=cols, how='left')
    oof_test.drop(cols, axis=1, inplace=True)
    oof.drop(cols, axis=1, inplace=True)
    return oof, oof_test

