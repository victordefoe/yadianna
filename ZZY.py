# encoding: utf-8
'''
@Author: 赵致远
@Time: 2019/9/27 0:03
@Contact: victordefoe88@gmail.com

@File: ZZY.py
@Statement:这一份代码是我的队友赵致远的工作，他和我组队参与的，因为参加得比较随意，也没什么分工，我俩基本是各自做
，很多工作是相同的。

'''

# coding: utf-8

# # Powered by tuzixini
# # 导入数据 查看数据基础信息

# In[1]:


import pandas as pd
import numpy as np
import pdb

pf = pd.read_csv('data/1/train/train_profile.csv')
bs = pd.read_csv('data/1/train/train_bankStatement.csv')
cb = pd.read_csv('data/1/train/train_creditBill.csv')
bh = pd.read_csv('data/1/train/train_behaviors.csv')
label = pd.read_csv('data/1/train/train_label.csv')

A_pf = pd.read_csv('data/1/A/test_profile_A.csv')
A_bs = pd.read_csv('data/1/A/test_bankStatement_A.csv')
A_cb = pd.read_csv('data/1/A/test_creditBill_A.csv')
A_bh = pd.read_csv('data/1/A/test_behaviors_A.csv')

# 查看基础信息
print('*****用户基本信息*****')
pf.info()
print('*****银行卡流水*****')
bs.info()
print('*****信用卡账单*****')
cb.info()
print('*****行为记录*****')
bh.info()
print('*****逾期标签*****')
label.info()

# #  处理表格数据

# In[2]:


# 几种归一化方式
# 集中数据到中值附近 -1->1
z_score_scaler = lambda x: (x - np.mean(x)) / (np.std(x))
# 按照最大值和最小值归一化分布 0->1
max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
sign_keep_scaler = lambda x: np.sign(x) * (np.abs(x) / (np.max(x) - np.min(x) + 1))


# 定义处理四个表格的函数

# 处理表 bf 的函数
def pf_per(pf):
    # one-hot 编码数据
    pf_ = pf['用户标识']
    temp = pd.get_dummies(pf['性别'])
    pf_ = pd.concat([pf_, temp], axis=1)
    temp = pd.get_dummies(pf['职业'])
    pf_ = pd.concat([pf_, temp], axis=1)
    temp = pd.get_dummies(pf['教育程度'])
    pf_ = pd.concat([pf_, temp], axis=1)
    temp = pd.get_dummies(pf['婚姻状态'])
    pf_ = pd.concat([pf_, temp], axis=1)
    temp = pd.get_dummies(pf['户口类型'])
    pf_ = pd.concat([pf_, temp], axis=1)
    pf_.columns = ['用户标识', '性别0', '性别1', '性别2', '职业0', '职业1', '职业2',
                   '职业3', '职业4', '教育0', '教育1', '教育2', '教育3', '教育4', '婚姻0',
                   '婚姻1', '婚姻2', '婚姻3', '婚姻4', '婚姻5', '户口0', '户口1', '户口2', '户口3', '户口4']
    return pf_


# In[3]:


# 用来处理表 bs 的函数
def bs_per(bs, uid, N=False, X=-1):
    # 暂定 为每个用户增加如下几列特征
    # 相同时间戳的交易认为发生在同一天

    # 1.流水余额类型  类型 0/1 对应 交易类型0/1  没有数据的用户此项填-1
    # 2.流水余额金额  类型0总金额 - 类型1总金额  没有数据的用户此项-1
    # 3.交易类型0 总频次   没有数据的用户填-1
    # 4.交易类型1 总频次   没有数据的用户填-1
    # 5.工资0次数  没有数据的用户填-1
    # 6.工资1次数
    # 7.工资0总金额   没有数据的用户填-1
    # 8.工资1总金额
    # 9.记录条数   没有数据的用户填-1

    ## TODO： 利用时间戳   交易类型0总金额 交易类型1总金额

    # 输入是bs, pf表的完整 用户uid 列
    # 输出是以用户uid在 第一列 为唯一标识 的补充完整的7列表格
    # uid 包含了所有用户的id
    # uid_ 包含了有数据的用户的id

    # 添加 用户标识
    bs_ = pd.DataFrame(uid)

    # 计算 流水余额  流水余额类型
    a = (bs['交易类型'] * 2 - 1) * bs['交易金额']
    a = pd.concat([bs['用户标识'], a], axis=1)
    a = pd.DataFrame(a.groupby('用户标识').sum().reset_index(level=['用户标识']))
    a.columns = ['用户标识', '流水余额']
    a['流水余额类型'] = a['流水余额'] / (a['流水余额'].abs())
    a['流水余额类型'] = (a['流水余额类型'] + 1) / 2
    a['流水余额'] = a['流水余额'].abs()
    # 归一化  流水余额
    if N:
        a['流水余额'] = a[['流水余额']].apply(N)
    uid_ = a['用户标识']
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 交易0频次
    a = pd.DataFrame(bs.groupby('用户标识')['交易类型'].value_counts())
    a = a.loc[(slice(None), 0), :]
    a = pd.DataFrame(a.reset_index(level=['用户标识']).values)
    a.columns = ['用户标识', '交易0频次']
    a = pd.merge(pd.DataFrame(uid_), a, on='用户标识', how='left')
    a = a.fillna(0)
    # 归一化 交易0频次
    if N:
        a['交易0频次'] = a[['交易0频次']].apply(N)
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 交易1频次
    a = pd.DataFrame(bs.groupby('用户标识')['交易类型'].value_counts())
    a = a.loc[(slice(None), 1), :]
    a = pd.DataFrame(a.reset_index(level=['用户标识']).values)
    a.columns = ['用户标识', '交易1频次']
    a = pd.merge(pd.DataFrame(uid_), a, on='用户标识', how='left')
    a = a.fillna(0)
    # 归一化 交易1频次
    if N:
        a['交易1频次'] = a[['交易1频次']].apply(N)
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 工资0次数
    a = pd.DataFrame(bs.groupby('用户标识')['工资收入标记'].value_counts())
    a = a.loc[(slice(None), 0), :]
    a = pd.DataFrame(a.reset_index(level=['用户标识']).values)
    a.columns = ['用户标识', '工资0次数']
    a = pd.merge(pd.DataFrame(uid_), a, on='用户标识', how='left')
    a = a.fillna(0)
    # 归一化 工资0次数
    if N:
        a['工资0次数'] = a[['工资0次数']].apply(N)
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 工资1次数
    a = pd.DataFrame(bs.groupby('用户标识')['工资收入标记'].value_counts())
    a = a.loc[(slice(None), 1), :]
    a = pd.DataFrame(a.reset_index(level=['用户标识']).values)
    a.columns = ['用户标识', '工资1次数']
    a = pd.merge(pd.DataFrame(uid_), a, on='用户标识', how='left')
    a = a.fillna(0)
    # 归一化 工资1次数
    if N:
        a['工资1次数'] = a[['工资1次数']].apply(N)
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 工资0金额
    a = (-1 * (bs['工资收入标记'] - 1)) * bs['交易金额']
    a = pd.concat([bs['用户标识'], a], axis=1)
    a = pd.DataFrame(a.groupby('用户标识').sum().reset_index(level=['用户标识']))
    a.columns = ['用户标识', '工资0金额']
    a = pd.merge(pd.DataFrame(uid_), a, on='用户标识', how='left')
    a = a.fillna(0)
    # 归一化 工资0金额
    if N:
        a['工资0金额'] = a[['工资0金额']].apply(N)
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 工资1金额
    a = bs['工资收入标记'] * bs['交易金额']
    a = pd.concat([bs['用户标识'], a], axis=1)
    a = pd.DataFrame(a.groupby('用户标识').sum().reset_index(level=['用户标识']))
    a.columns = ['用户标识', '工资1金额']
    a = pd.merge(pd.DataFrame(uid_), a, on='用户标识', how='left')
    a = a.fillna(0)
    # 归一化 工资1金额
    if N:
        a['工资1金额'] = a[['工资1金额']].apply(N)
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 计算 记录条数
    a = bs_['交易0频次'] + bs_['交易1频次']
    a = pd.DataFrame(a)
    a = pd.concat([bs_['用户标识'], a], axis=1)
    a.columns = ['用户标识', '交易总次数']
    bs_ = pd.merge(bs_, a, on='用户标识', how='left')

    # 填充NaN
    bs_ = bs_.fillna(X)
    return bs_


# In[12]:


# 用来处理表 cb 的函数
def cb_per(bs, uid, N=False, X=-1):
    # uid 是全部用户的id列表
    uid = pd.DataFrame(uid)  # 64436 rows × 1 columns
    # cb_users 是cb 表的用户列
    cb_users = pd.DataFrame(cb['用户标识'])  # 937836 rows × 1 columns
    cb_ = uid

    # 删除时间戳为0的数据
    cb.drop(cb[cb.账单时间戳 == 0].index, inplace=True)

    # 处理 银行标识,  -- 结果为多热码
    a = cb.drop_duplicates(['用户标识', '银行标识'], keep='first', inplace=False)  # 114779
    banks = pd.get_dummies(a['银行标识'], prefix='银行')  # 114779
    # pdb.set_trace()
    a = pd.concat([cb_users, banks], axis=1).groupby(['用户标识']).sum().reset_index()
    cb_ = pd.merge(cb_, a, on='用户标识', how='left')

    # 处理 上期账单金额


    # 处理 上期还款金额


    # 处理 信用卡额度
    limit = pd.DataFrame(cb.drop_duplicates(['用户标识', '银行标识'], keep='first', inplace=False)['信用卡额度'])  # 114779
    if N:
        limit['信用卡额度'] = limit[['信用卡额度']].apply(N)
    a = banks.multiply(limit['信用卡额度'], axis=0).rename(columns=lambda x: '额度_' + x)
    a = pd.concat([cb_users, a], axis=1).groupby(['用户标识']).sum().reset_index()
    cb_ = pd.merge(cb_, a, on='用户标识', how='left')

    a = pd.DataFrame(pd.concat([a['用户标识'], a[a.columns[1:13]].sum(axis=1)], axis=1))
    a.columns = ['用户标识', '总额度']
    if N:
        a['总额度'] = a[['总额度']].apply(N)
    cb_ = pd.merge(cb_, a, on='用户标识', how='left')
    # 处理 本期账单余额

    # # 处理 还款状态
    a = cb.groupby(['用户标识', '银行标识'])['还款状态'].max().reset_index().drop('银行标识', axis=1).groupby(
        ['用户标识']).sum().reset_index()
    cb_ = pd.merge(cb_, a, on='用户标识', how='left')
    # print(cb_.info())
    # 处理信用卡额度变化

    # 处理额度比一开始办卡时候增减了多少-每位用户每张卡的额度是增加还是减少，相比于一开始办卡时？
    dif = cb.groupby(['用户标识', '银行标识'])['信用卡额度'].max() - cb.groupby(['用户标识', '银行标识'])['信用卡额度'].min()
    dif[dif.values > 0].count()  # 一共有这么多用户的信用卡额度发生了变化
    change = cb.groupby(['用户标识', '银行标识'])['信用卡额度'].last() - cb.groupby(['用户标识', '银行标识'])['信用卡额度'].first()
    card_chg = pd.DataFrame(change)
    card_chg.columns = ['额度变化']
    card_chg = card_chg.reset_index()
    a = card_chg['额度变化'].values
    users = pd.DataFrame(cb.drop_duplicates(['用户标识', '银行标识'], keep='first', inplace=False))['用户标识']
    card_chg = pd.concat([users, banks], axis=1).drop(['用户标识'], axis=1).multiply(a, axis=0).rename(
        columns=lambda x: '额度变化_' + x)
    # 将每一个人不同的银行卡额度变化取和
    card_chg = pd.concat([users, card_chg], axis=1).groupby(['用户标识']).sum().reset_index()
    card_chg_ = card_chg.copy()
    if N:
        card_chg = card_chg.apply(N)
    cb_ = pd.merge(cb_, card_chg, on='用户标识', how='left')

    # 额度变化 符号
    b = card_chg_.drop(['用户标识'], axis=1).rename(columns=lambda x: '额度变化_' + x + '_符号')
    # b = pd.concat([users, banks], axis=1).drop(['用户标识'],axis=1).multiply(b, axis=0).rename(columns=lambda x: '额度变化_'+ x+'_符号')
    b = b.apply(lambda x: x / x.abs())
    b = b.apply(lambda x: (x + 1) / 2)
    b = pd.concat([card_chg_['用户标识'], b], axis=1).groupby(['用户标识']).apply(lambda x: x).reset_index()
    cb_ = pd.merge(cb_, b.drop(['index'], axis=1), on='用户标识', how='left')

    # # 填充NaN
    cb_ = cb_.fillna(X)

    return cb_


# In[18]:


# 用来处理表 bh 的函数
def bh_per(bh, uid, X=-1):
    bhdata = pd.read_csv('bh_lc.csv')
    bhdata = bhdata.drop(['Unnamed: 0'], axis=1)
    bhdata = bhdata.rename(columns={'id': '用户标识'})
    bh_ = pd.merge(pd.DataFrame(uid), bhdata, on='用户标识', how='left')
    bh_ = bh_.fillna(X)
    return bh_


# data = ClearData(pf,bs,cb,bh,pri=False)


# In[6]:


# 集合四个表清理数据过程的函数
def ClearData(pf, bs, cb, bh, pri=True):
    pf_ = pf_per(pf)
    bs_ = bs_per(bs, pf['用户标识'], N=max_min_scaler)
    cb_ = cb_per(cb, pf['用户标识'], N=max_min_scaler)
    bh_ = bh_per(bh, pf['用户标识'])
    if pri:
        print(pf_.info())
        print(bs_.info())
        print(cb_.info())
        print(bh_.info())
    data = pd.merge(pf_, bs_, on='用户标识', how='left')
    data = pd.merge(data, cb_, on='用户标识', how='left')
    data = pd.merge(data, bh_, on='用户标识', how='left')
    # TODO: 附加剩下两张表的数据
    return data


# # 计算准确率的函数

# In[20]:


# 计算准确率的函数
import scipy as sp
from scipy import stats
import pdb


def ks_calc_2samp(pred, label):
    data = pd.DataFrame()
    data[0] = pred
    data[1] = label
    data.columns = ['pred', 'label']
    class_col = ['label'];
    score_col = ['pred']
    # print(data.head())
    # pdb.set_trace()
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


# ks_2samp, cdf_2samp = ks_calc_2samp(pred, label)


# # 建模训练测试

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
import warnings

warnings.filterwarnings('ignore')


# 定义用来训练的函数
def train(X, y):
    # 定义模型
    clf = XGBClassifier(n_jobs=6, learning_rate=0.1, n_estimator=50, max_depth=6, silent=False,
                        objective='binary:logistic')
    param_test = {'n_estimators': [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],
                  'max_depth': [2, 3, 4, 5, 6, 7]}
    grid_search = GridSearchCV(estimator=clf, param_grid=param_test, scoring='accuracy', cv=5, verbose=1)
    model = clf
    # model = grid_search

    model.fit(X, y)
    return model


def test(model, X):
    y = model.predict_proba(X)[:, 1]
    return y


# K-S 测试
def ks(a, b):
    result, a_ = ks_calc_2samp(a, b)
    print('KS')
    print(result)
    return result


# 获取数据
val = True

bs_all = pd.concat([bs, A_bs], axis=0)
bh_all = pd.concat([bh, A_bh], axis=0)
cb_all = pd.concat([cb, A_cb], axis=0)
pf_all = pd.concat([pf, A_pf], axis=0)
data_all = ClearData(pf_all, bs_all, cb_all, bh_all, pri=False)
# data_all.to_csv('./featurezzy.csv')


# 训练,验证,测试,生成结果
if val:
    data = data_all[np.isin(data_all.用户标识, pf['用户标识'].values)]
    data.drop(['用户标识'], axis=1, inplace=True)

    X, X_val, y, y_val = train_test_split(data, label['标签'], test_size=0.2, random_state=0)
    model = train(X, y)
    y_ = test(model, X_val)
    y_ = pd.Series(y_).reset_index(drop=True)
    # print(y_)
    y_val = pd.Series(y_val).reset_index(drop=True)
    res = ks(y_, y_val.values)
    auc = metrics.roc_auc_score(y_val.values, y_)
    print('AUC')
    print(auc)
else:
    A_data = data_all[np.isin(data_all.用户标识, A_pf['用户标识'].values)]
    A_data.drop(['用户标识'], axis=1, inplace=True)

    X = sets[np.isin(data_all.id, pf['用户标识'].values)]
    y = label['标签']
    model = train(X, y)
    A_pro = test(model, A_data)
    probs = pd.Series(A_pro)
    upload = pd.concat([A_id, probs], axis=1)
    upload.to_csv('./upload_zzy.csv', header=0, index=0)

# In[ ]:


y_val

# In[ ]:


res = ks(y_, y_val.values)
print(res)
print(auc)

# In[28]:


pd.read_csv('./out/upload.csv', header=None)

