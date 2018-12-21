# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:31:49 2018

@author: shaowu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import gc
from collections import Counter
#seed=2018
#np.random.seed(seed)
def count_feat(data):return data.shape[0]
 
def xgb_model(X_train,y_train,X_test,y_test):
    '''定义xgb模型'''
    dtrain=xgb.DMatrix(X_train,y_train)
    dval=xgb.DMatrix(X_test,y_test)
    num_rounds=50000  #迭代次数
    params={'booster':'gbtree',
            'eta':0.1, #学习率
            'max_depth':5, #树的深度
            'subsample':0.8,
            'colsample_bytree':0.8,
            'objective':'binary:logistic',
            'eval_metric': 'auc',# 'logloss',
            'random_seed':2018 #随机种子
            }
    watchlist = [(dtrain,'train'),(dval,'val')]
    ####模型训练：
    model=xgb.train(params,dtrain,num_rounds,watchlist,verbose_eval=100,early_stopping_rounds=100)

    return model

train_path_chu='train_chu/'
train_path='train_fu/'
test_path='test/'
#-----------------------------读入初始训练数据---------------------------------

train1= pd.read_csv(open(train_path+'法人行政许可注（撤、吊）销信息.csv',encoding='utf8',errors='ignore'))
train1_chu= pd.read_csv(open(train_path_chu+'法人行政许可注（撤、吊）销信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train1['企业名称'] if i in train1_chu['企业名称']]))
train1=pd.concat([train1,train1_chu],axis=0).reset_index(drop=True)
print('去重前',len(train1),len(set(train1['企业名称'])))
train1=train1.drop_duplicates().reset_index(drop=True)
print('去重后',len(train1))
del train1_chu
train2= pd.read_csv(open(train_path+'分支机构信息.csv',encoding='utf8',errors='ignore'))
train2_chu= pd.read_csv(open(train_path_chu+'分支机构信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train2['企业名称'] if i in train2_chu['企业名称']]))
train2=pd.concat([train2,train2_chu],axis=0).reset_index(drop=True)
print('去重前',len(train2),len(set(train2['企业名称'])))
train2=train2.drop_duplicates().reset_index(drop=True)
print('去重后',len(train2))
del train2_chu
train3= pd.read_csv(open(train_path+'机构设立（变更）登记信息.csv',encoding='utf8',errors='ignore'))
train3_chu= pd.read_csv(open(train_path_chu+'机构设立（变更）登记信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train3['企业名称'] if i in train3_chu['企业名称']]))
train3=pd.concat([train3,train3_chu],axis=0).reset_index(drop=True)
print('去重前',len(train3),len(set(train3['企业名称'])))
train3=train3.drop_duplicates().reset_index(drop=True)
print('去重后',len(train3))
del train3_chu
train4=pd.read_csv(open(train_path+'企业表彰荣誉信息.csv',encoding='utf8',errors='ignore'))
train4_chu=pd.read_csv(open(train_path_chu+'企业表彰荣誉信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train4['企业名称'] if i in train4_chu['企业名称']]))
train4=pd.concat([train4,train4_chu],axis=0).reset_index(drop=True)
print('去重前',len(train4),len(set(train4['企业名称'])))
train4=train4.drop_duplicates().reset_index(drop=True)
print('去重后',len(train4))
del train4_chu
train5=pd.read_csv(open(train_path+'企业非正常户认定.csv',encoding='utf8',errors='ignore'))
train5_chu=pd.read_csv(open(train_path_chu+'企业非正常户认定.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train5['企业名称'] if i in train5_chu['企业名称']]))
train5=pd.concat([train5,train5_chu],axis=0).reset_index(drop=True)
print('去重前',len(train5),len(set(train5['企业名称'])))
train5=train5.drop_duplicates().reset_index(drop=True)
print('去重后',len(train5))
del train5_chu
train6=pd.read_csv(open(train_path+'企业基本信息&高管信息&投资信息.csv',encoding='utf8',errors='ignore'))
train6_chu=pd.read_csv(open(train_path_chu+'企业基本信息&高管信息&投资信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train6['企业名称'] if i in train6_chu['企业名称']]))
train6=pd.concat([train6,train6_chu],axis=0).reset_index(drop=True)
print('去重前',len(train6),len(set(train6['企业名称'])))
train6=train6.drop_duplicates().reset_index(drop=True)
print('去重后',len(train6))
del train6_chu
train7=pd.read_csv(open(train_path+'企业税务登记信息.csv',encoding='utf8',errors='ignore'))
train7_chu=pd.read_csv(open(train_path_chu+'企业税务登记信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train7['企业名称'] if i in train7_chu['企业名称']]))
train7=pd.concat([train7,train7_chu],axis=0).reset_index(drop=True)
print('去重前',len(train7),len(set(train7['企业名称'])))
train7=train7.drop_duplicates().reset_index(drop=True)
print('去重后',len(train7))
del train7_chu
train8=pd.read_csv(open(train_path+'双打办打击侵权假冒处罚案件信息.csv',encoding='utf8',errors='ignore'))
train8_chu=pd.read_csv(open(train_path_chu+'双打办打击侵权假冒处罚案件信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train8['企业名称'] if i in train8_chu['企业名称']]))
train8=pd.concat([train8,train8_chu],axis=0).reset_index(drop=True)
print('去重前',len(train8),len(set(train8['企业名称'])))
train8=train8.drop_duplicates().reset_index(drop=True)
print('去重后',len(train8))
del train8_chu
train9=pd.read_csv(open(train_path+'双公示-法人行政许可信息.csv',encoding='utf8',errors='ignore'))
train9_chu=pd.read_csv(open(train_path_chu+'双公示-法人行政许可信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train9['企业名称'] if i in train9_chu['企业名称']]))
train9=pd.concat([train9,train9_chu],axis=0).reset_index(drop=True)
print('去重前',len(train9),len(set(train9['企业名称'])))
train9=train9.drop_duplicates().reset_index(drop=True)
print('去重后',len(train9))
del train9_chu
train10=pd.read_csv(open(train_path+'许可资质年检信息.csv',encoding='utf8',errors='ignore'))
train10_chu=pd.read_csv(open(train_path_chu+'许可资质年检信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train10['企业名称'] if i in train10_chu['企业名称']]))
train10=pd.concat([train10,train10_chu],axis=0).reset_index(drop=True)
print('去重前',len(train10),len(set(train10['企业名称'])))
train10=train10.drop_duplicates().reset_index(drop=True)
print('去重后',len(train10))
del train10_chu
train11=pd.read_csv(open(train_path+'招聘数据.csv',encoding='utf8',errors='ignore'))
train11_chu=pd.read_csv(open(train_path_chu+'招聘数据.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train11['企业名称'] if i in train11_chu['企业名称']]))
train11=pd.concat([train11,train11_chu],axis=0).reset_index(drop=True)
print('去重前',len(train11),len(set(train11['企业名称'])))
train11=train11.drop_duplicates().reset_index(drop=True)
print('去重后',len(train11))
del train11_chu
train12=pd.read_csv(open(train_path+'资质登记（变更）信息.csv',encoding='utf8',errors='ignore'))
train12_chu=pd.read_csv(open(train_path_chu+'资质登记（变更）信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in train12['企业名称'] if i in train12_chu['企业名称']]))
train12=pd.concat([train12,train12_chu],axis=0).reset_index(drop=True)
print('去重前',len(train12),len(set(train12['企业名称'])))
train12=train12.drop_duplicates().reset_index(drop=True)
print('去重后',len(train12))
del train12_chu
label11=pd.read_csv(open(train_path+'失信被执行人名单.csv',encoding='utf8',errors='ignore'))
label11_chu=pd.read_csv(open(train_path_chu+'失信被执行人名单.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in label11['企业名称'] if i in label11_chu['企业名称']]))
#label1['label1']=1
label22=pd.read_csv(open(train_path+'双公示-法人行政处罚信息.csv',encoding='utf8',errors='ignore'))
label22_chu=pd.read_csv(open(train_path_chu+'双公示-法人行政处罚信息.csv',encoding='utf8',errors='ignore'))
print('********\n',len([i for i in label22['企业名称'] if i in label22_chu['企业名称']]))
#label2['label2']=1
print(len(label11),len(set(label11['企业名称'])),\
      len(label22),len(set(label22['企业名称'])))
label1=pd.DataFrame(list(set(label11['企业名称'])),columns=['企业名称'])
label1=pd.concat([label1,label11_chu],axis=0).reset_index(drop=True)
label1['label1']=1
print('去重前',len(label1))
label1=label1.drop_duplicates().reset_index(drop=True)
print('去重后',len(label1))
label2=pd.DataFrame(list(set(label22['企业名称'])),columns=['企业名称'])
label2=pd.concat([label2,label22_chu],axis=0).reset_index(drop=True)
label2['label2']=1
print('去重前',len(label2))
label2=label2.drop_duplicates().reset_index(drop=True)
print('去重后',len(label2))
del label11,label11_chu,label22,label22_chu
print('读入训练数据完毕！\n读入测试数据...')
#-----------------------------读入初始测试数据---------------------------------
test1= pd.read_csv(open(test_path+'法人行政许可注（撤、吊）销信息.csv',encoding='utf8',errors='ignore'))
test2= pd.read_csv(open(test_path+'分支机构信息.csv',encoding='utf8',errors='ignore'))
test3= pd.read_csv(open(test_path+'机构设立（变更）登记信息.csv',encoding='utf8',errors='ignore'))
test4=pd.read_csv(open(test_path+'企业表彰荣誉信息.csv',encoding='utf8',errors='ignore'))
test5=pd.read_csv(open(test_path+'企业非正常户认定.csv',encoding='utf8',errors='ignore'))
test6=pd.read_csv(open(test_path+'企业基本信息&高管信息&投资信息.csv',encoding='utf8',errors='ignore'))
test7=pd.read_csv(open(test_path+'企业税务登记信息.csv',encoding='utf8',errors='ignore'))
test8=pd.read_csv(open(test_path+'双打办打击侵权假冒处罚案件信息.csv',encoding='utf8',errors='ignore'))
test9=pd.read_csv(open(test_path+'双公示-法人行政许可信息.csv',encoding='utf8',errors='ignore'))
test10=pd.read_csv(open(test_path+'许可资质年检信息.csv',encoding='utf8',errors='ignore'))
test11=pd.read_csv(open(test_path+'招聘数据.csv',encoding='utf8',errors='ignore'))
test12=pd.read_csv(open(test_path+'资质登记（变更）信息.csv',encoding='utf8',errors='ignore'))

#------------------------训练集企业名称、测试集企业名称----------------------------
train_id=pd.DataFrame(list(set(train6['企业名称'])),columns=['企业名称']) # 28365
test_id=pd.DataFrame(list(set(test6['企业名称'])),columns=['企业名称'])  # 17534
train_id=pd.merge(train_id,label1,how='left',on=['企业名称']).fillna(0)
train_id=pd.merge(train_id,label2,how='left',on=['企业名称']).fillna(0)
#alldata_id=list(set(alldata6['企业名称']))  # 45899
#print(len(train_id),len(test_id),len(alldata_id))

#-------------------------训练集与测试集结合，方便提取特征-----------------------
alldata1=pd.concat([train1,test1],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata1),len(set(alldata1['企业名称'])))
alldata2=pd.concat([train2,test2],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata2),len(set(alldata2['企业名称'])))
alldata3=pd.concat([train3,test3],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata3),len(set(alldata3['企业名称'])))
alldata4=pd.concat([train4,test4],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata4),len(set(alldata4['企业名称'])))
alldata5=pd.concat([train5,test5],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata5),len(set(alldata5['企业名称'])))
alldata6=pd.concat([train6,test6],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata6),len(set(alldata6['企业名称'])))
alldata7=pd.concat([train7,test7],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata7),len(set(alldata7['企业名称'])))
alldata8=pd.concat([train8,test8],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata8),len(set(alldata8['企业名称'])))
alldata9=pd.concat([train9,test9],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata9),len(set(alldata9['企业名称'])))
alldata10=pd.concat([train10,test10],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata10),len(set(alldata10['企业名称'])))
alldata11=pd.concat([train11,test11],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata11),len(set(alldata11['企业名称'])))
alldata12=pd.concat([train12,test12],axis=0).reset_index(drop=True).fillna(-99)
print(len(alldata12),len(set(alldata12['企业名称'])))

def one_hot_col(col):
    from sklearn import preprocessing
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl

col='关联机构设立登记表主键ID'
ll=one_hot_col(\
list(alldata1[col])+list(alldata4[col])+list(alldata5[col])+list(alldata7[col])+\
list(alldata8[col])+list(alldata9[col])+list(alldata10[col])+list(alldata12[col]))

alldata1[col],alldata4[col],alldata5[col],alldata7[col],alldata8[col],\
alldata9[col],alldata10[col],alldata12[col]=ll.transform(list(alldata1[col])),\
ll.transform(list(alldata4[col])),ll.transform(list(alldata5[col])),\
ll.transform(list(alldata7[col])),ll.transform(list(alldata8[col])),\
ll.transform(list(alldata9[col])),ll.transform(list(alldata10[col])),\
ll.transform(list(alldata12[col]))
col='工商注册号'
ll=one_hot_col(\
list(alldata1[col])+list(alldata3[col])+list(alldata4[col])+list(alldata7[col])+\
list(alldata8['被处罚企业工商注册号'])+list(alldata10[col])+list(alldata12[col]))

alldata1[col],alldata3[col],alldata4[col],alldata7[col],alldata8[col],alldata10[col],\
alldata12[col]=ll.transform(list(alldata1[col])),\
ll.transform(list(alldata3[col])),ll.transform(list(alldata4[col])),\
ll.transform(list(alldata7[col])),ll.transform(list(alldata8['被处罚企业工商注册号'])),\
ll.transform(list(alldata10[col])),ll.transform(list(alldata12[col]))
col='信息提供部门编码'
ll=one_hot_col(\
list(alldata1[col])+list(alldata3[col])+list(alldata4[col])+list(alldata5[col])+\
list(alldata7[col])+list(alldata8[col])+list(alldata9[col])+list(alldata10[col])+\
list(alldata12[col]))

alldata1[col],alldata3[col],alldata4[col],alldata5[col],alldata7[col],\
alldata8[col],alldata9[col],alldata10[col],alldata12[col]=ll.transform(\
        list(alldata1[col])),\
ll.transform(list(alldata3[col])),ll.transform(list(alldata4[col])),\
ll.transform(list(alldata5[col])),ll.transform(list(alldata7[col])),\
ll.transform(list(alldata8[col])),ll.transform(list(alldata9[col])),\
ll.transform(list(alldata10[col])),ll.transform(list(alldata12[col]))
col='组织机构代码'
ll=one_hot_col(\
list(alldata1[col])+list(alldata3[col])+list(alldata4[col])+list(alldata5[col])+\
list(alldata7[col])+list(alldata10[col])+\
list(alldata12[col]))

alldata1[col],alldata3[col],alldata4[col],alldata5[col],alldata7[col],\
alldata10[col],alldata12[col]=ll.transform(\
        list(alldata1[col])),\
ll.transform(list(alldata3[col])),ll.transform(list(alldata4[col])),\
ll.transform(list(alldata5[col])),ll.transform(list(alldata7[col])),\
ll.transform(list(alldata10[col])),ll.transform(list(alldata12[col]))

col='任务编号'
ll=one_hot_col(\
list(alldata1[col])+list(alldata3[col])+list(alldata4[col])+list(alldata5[col])+\
list(alldata7[col])+list(alldata8[col])+list(alldata9[col])+list(alldata10[col])+\
list(alldata12[col]))

alldata1[col],alldata3[col],alldata4[col],alldata5[col],alldata7[col],\
alldata8[col],alldata9[col],alldata10[col],alldata12[col]=ll.transform(\
        list(alldata1[col])),\
ll.transform(list(alldata3[col])),ll.transform(list(alldata4[col])),\
ll.transform(list(alldata5[col])),ll.transform(list(alldata7[col])),\
ll.transform(list(alldata8[col])),ll.transform(list(alldata9[col])),\
ll.transform(list(alldata10[col])),ll.transform(list(alldata12[col]))

col='统一社会信用代码'
ll=one_hot_col(\
list(alldata1[col])+list(alldata3[col])+list(alldata4[col])+list(alldata6[col])+\
list(alldata7[col])+list(alldata8['被处罚企业统一社会信用编码'])+list(alldata10[col])+\
list(alldata12[col]))

alldata1[col],alldata3[col],alldata4[col],alldata6[col],alldata7[col],\
alldata8[col],alldata10[col],alldata12[col]=ll.transform(\
        list(alldata1[col])),\
ll.transform(list(alldata3[col])),ll.transform(list(alldata4[col])),\
ll.transform(list(alldata6[col])),ll.transform(list(alldata7[col])),\
ll.transform(list(alldata8['被处罚企业统一社会信用编码'])),\
ll.transform(list(alldata10[col])),ll.transform(list(alldata12[col]))
col='企业名称'
print('所有企业个数',len(set(list(alldata1[col])+list(alldata2[col])+\
list(alldata3[col])+list(alldata4[col])+list(alldata5[col])+list(alldata6[col])+\
list(alldata7[col])+list(alldata8[col])+list(alldata9[col])+list(alldata10[col])+\
list(alldata11[col])+list(alldata12[col]))))

#-----------------------------筛选特征,之后对这部分提取特征即可-----------------------------
print('筛选特征...')
feat1=alldata1[['企业名称','关联机构设立登记表主键ID','工商注册号','注（撤、吊）销原因',\
               '数据状态', '数据来源','信息提供部门编码','组织机构代码',\
               '统一社会信用代码','许可证编号','任务编号']]
feat2=alldata2[['企业名称','分支机构省份','分支机构状态','分支行业门类',\
               '分支行业代码','分支机构区县', '分支机构类型']]
feat3=alldata3[['企业名称','数据状态', '数据来源','信息提供部门编码','组织机构代码',\
               '统一社会信用代码','法定代表人姓名','法定代表人证件号码', '注册（开办）资金',\
               '企业类型代码','所属行业代码', '行政区划','工商注册号','种类',\
               '机构地址（住所）','企业经度','企业纬度','是否有经纬度','任务编号',\
               '经济类型','企业地址是否有变化']]
feat4=alldata4[['企业名称','关联机构设立登记表主键ID','数据状态', '数据来源',\
                '信息提供部门编码','组织机构代码','任务编号','工商注册号',\
               '统一社会信用代码','荣誉等级', '认定机关全称']]
feat5=alldata5[['企业名称','关联机构设立登记表主键ID','数据状态', '数据来源',\
                '信息提供部门编码','组织机构代码','任务编号','税务管理码','纳税人识别号',\
               '应用年限','登记注册类型','管理机构','纳税人状态']]
feat6=alldata6[['企业名称','注册号', '统一社会信用代码', '注册资金','企业(机构)类型名称',\
                '行业门类代码','住所所在地省份','姓名', '首席代表标志','职务',\
                '投资人','注册资本(金)币种名称']]
feat7=alldata7[['企业名称','关联机构设立登记表主键ID','数据状态', '数据来源',\
                '信息提供部门编码','组织机构代码','任务编号','工商注册号',\
                '登记注册类型','审核结果', '审核单位','区域',\
                '统一社会信用代码','税务管理码','纳税人识别号']]
feat8=alldata8[['企业名称','关联机构设立登记表主键ID','数据状态', '数据来源',\
                '信息提供部门编码','任务编号','统一社会信用代码',\
                '工商注册号','公布方式及网址']]
feat9=alldata9[['企业名称','关联机构设立登记表主键ID','数据状态_1', '数据来源',\
                '信息提供部门编码','任务编号','行政相对人代码_2','行政相对人代码_1',\
               '行政相对人代码＿3','项目名称','审批类别', '许可机关','地方编码',\
               '备注','数据状态_2']]
feat10=alldata10[['企业名称','关联机构设立登记表主键ID','数据状态', '数据来源',\
                '信息提供部门编码','任务编号','组织机构代码','年检结果',\
               '年检机关全称','年检事项名称','工商注册号','统一社会信用代码',\
               '证书编号','权力编码']]
feat11=alldata11[['企业名称','网站名称','工作经验', '工作地点',\
                '招聘人数','职位月薪','最低学历','业务主键']]
feat12=alldata12[['企业名称','关联机构设立登记表主键ID','数据状态', '数据来源',\
                '信息提供部门编码','任务编号','组织机构代码','工商注册号',\
                '统一社会信用代码','种类','认定机关全称','权力编码']]
def label_code(feat):
    '''对特征进行编码'''
    from sklearn import preprocessing
    label = preprocessing.LabelEncoder()
    k=1
    for col in feat.columns:
        if feat[col].dtype =='object' and col !='企业名称':
            print('第%d个编码特征:%s'%(k,col))
            feat[col] = label.fit_transform(feat[col].astype(str))
            k=k+1
    return feat

def count_col(data,col):
    '''
    data 待处理数据集
    col 待处理的字段名
    '''
    m=[]
    user_list=list(set(data['企业名称']))
    for user in user_list:
        mode_sub={}
        user_data=data[data['企业名称']==user]
        mode_list=list(user_data[col])
        while -99 in mode_list:
            mode_list.remove(-99)
        for i in mode_list:
            mode_sub[i]=mode_sub.get(i,0)+1
        m.append(mode_sub)
    m=pd.DataFrame(m)
    m['企业名称']=user_list
    return m

#------------------feat6特征提取----
#注册号频数：
per=feat6[['企业名称','注册号']].drop_duplicates().reset_index(drop=True)
per=per[per['注册号']!=-99].reset_index(drop=True)
print(len(per),len(set(per['注册号'])))
feat6_1=per.groupby(['企业名称'],as_index=False)['注册号'].agg({
        '注册号_count':'count'})
del per
#统一社会信用代码： 一对多的关系
per=feat6[['企业名称','统一社会信用代码']].drop_duplicates().reset_index(drop=True)
per=per[per['统一社会信用代码']!=-99].reset_index(drop=True)
print(len(per),len(set(per['统一社会信用代码'])))
feat6_2=per.groupby(['企业名称'],as_index=False)['统一社会信用代码'].agg({
        '统一社会信用代码_count':'count'}) #频数做特征
feat6_3=per # 直接作为特征，多了17条样本
del per
#注册资本(金)币种名称
per=feat6[['企业名称','注册资本(金)币种名称']].drop_duplicates().reset_index(drop=True)
per=per[per['注册资本(金)币种名称']!=-99].reset_index(drop=True)
print(len(per),len(set(per['注册资本(金)币种名称'])))
feat6_4=per.groupby(['企业名称'],as_index=False)['注册资本(金)币种名称'].agg({
        '注册资本(金)币种名称_count':'count'})
del per
#企业(机构)类型名称
per=feat6[['企业名称','企业(机构)类型名称']].drop_duplicates().reset_index(drop=True)
per=per[per['企业(机构)类型名称']!=-99].reset_index(drop=True)
print(len(per),len(set(per['企业(机构)类型名称'])))
feat6_5=per.groupby(['企业名称'],as_index=False)['企业(机构)类型名称'].agg({
        '企业(机构)类型名称_count':'count'}) #频数
import os
if not os.path.exists('feat6_6.csv'):
    feat6_6=count_col(per,'企业(机构)类型名称').fillna(0)
    #feat6_6=feat6_6.fillna(0)
    feat6_6.to_csv('feat6_6.csv',index=None,encoding='utf8')
else:
    feat6_6=pd.read_csv('feat6_6.csv',encoding='utf8')
del per
#行业门类代码
per=feat6[['企业名称','行业门类代码']].drop_duplicates().reset_index(drop=True)
per=per[per['行业门类代码']!=-99].reset_index(drop=True)
print(len(per),len(set(per['行业门类代码'])))
feat6_7=per.groupby(['企业名称'],as_index=False)['行业门类代码'].agg({
        '行业门类代码_count':'count'}) #频数
if not os.path.exists('feat6_8.csv'):
    feat6_8=count_col(per,'行业门类代码').fillna(0)
    #feat6_6=feat6_6.fillna(0)
    feat6_8.to_csv('feat6_8.csv',index=None,encoding='utf8')
else:
    feat6_8=pd.read_csv('feat6_8.csv',encoding='utf8')
del per
#住所所在地省份
per=feat6[['企业名称','住所所在地省份']].drop_duplicates().reset_index(drop=True)
per=per[per['住所所在地省份']!=-99].reset_index(drop=True)
print(len(per),len(set(per['住所所在地省份'])))
feat6_9=per.groupby(['企业名称'],as_index=False)['住所所在地省份'].agg({
        '住所所在地省份_count':'count'}) #频数
if not os.path.exists('feat6_10.csv'):
    feat6_10=count_col(per,'住所所在地省份').fillna(0)
    #feat6_6=feat6_6.fillna(0)
    feat6_10.to_csv('feat6_10.csv',index=None,encoding='utf8')
else:
    feat6_10=pd.read_csv('feat6_10.csv',encoding='utf8')
del per
#首席代表标志
per=feat6[['企业名称','首席代表标志']].drop_duplicates().reset_index(drop=True)
per=per[per['首席代表标志']!=-99].reset_index(drop=True)
print(len(per),len(set(per['首席代表标志'])))
if not os.path.exists('feat6_11.csv'):
    feat6_11=count_col(per,'首席代表标志').fillna(0)
    feat6_11.to_csv('feat6_11.csv',index=None,encoding='utf8')
else:
    feat6_11=pd.read_csv('feat6_11.csv',encoding='utf8')
del per
#职务
per=feat6[['企业名称','职务']].drop_duplicates().reset_index(drop=True)
per=per[per['职务']!=-99].reset_index(drop=True)
print(len(per),len(set(per['职务'])))
if not os.path.exists('feat6_12.csv'):
    feat6_12=count_col(per,'职务').fillna(0)
    feat6_12.to_csv('feat6_12.csv',index=None,encoding='utf8')
else:
    feat6_12=pd.read_csv('feat6_12.csv',encoding='utf8')
del per
#注册资本(金)币种名称
per=feat6[['企业名称','注册资本(金)币种名称']].drop_duplicates().reset_index(drop=True)
per=per[per['注册资本(金)币种名称']!=-99].reset_index(drop=True)
print(len(per),len(set(per['注册资本(金)币种名称'])))
if not os.path.exists('feat6_13.csv'):
    feat6_13=count_col(per,'注册资本(金)币种名称').fillna(0)
    feat6_13.to_csv('feat6_13.csv',index=None,encoding='utf8')
else:
    feat6_13=pd.read_csv('feat6_13.csv',encoding='utf8')
del per
#投资人频数
per=feat6[['企业名称','投资人']].drop_duplicates().reset_index(drop=True)
per=per[per['投资人']!=-99].reset_index(drop=True)
print(len(per),len(set(per['投资人'])))
feat6_14=per.groupby(['企业名称'],as_index=False)['投资人'].agg({
        '投资人_count':'count'}) #频数
del per
##姓名和投资人：
if not os.path.exists('feat6_15.csv'):
    feat6_15=[]
    per=feat6[['企业名称','姓名','投资人']].drop_duplicates().reset_index(drop=True)
    user_id=list(set(per['企业名称']))
    for i in user_id:
        m=per[per['企业名称']==i]
        user_list=set(list(m['姓名'])+list(m['投资人']))
        #user_list=list(set(user_list))
        if -99 in user_list:
            user_list.remove(-99)
        feat6_15.append([i,len(user_list)])
    feat6_15=pd.DataFrame(feat6_15,columns=['企业名称','姓名_投资人_num'])
    feat6_15.to_csv('feat6_15.csv',index=None,encoding='utf8')
else:
    feat6_15=pd.read_csv('feat6_15.csv',encoding='utf8')

#############
def count_num(col,alldata):
    '''
    定义计算频数函数
    col ---str字段
    alldata ---DataFrame数据
    return ---DataFrame频数数据
    '''
    m=dict()
    m_list=alldata[col].values
    n=len(alldata)
    for k,v in Counter(m_list).items():
        if k in m.keys():
            m[k] += v/n
        else:
            m[k] = v/n
    if -99 in m.keys(): #删除关键字-99
        del m[-99]
    return pd.DataFrame({col:list(m.keys()),\
                          col+'_num':list(m.values())})
#结合频数特征，并删除原始字段：
#对于alldata数据，计算same_cols中的频数：
def create_feat(col,data):
    '''
    定义构造特征函数
    col ---str字段
    data ---DataFrame数据
    return ---list列表特征
    k ---天数
    '''
    k=1
    if -99 in set(data[col]):
        m=list(set(data[col]))
        m.remove(-99)
        if len(m)>0:
            return [len(data)/k,len(m)/k,np.max(m)/k,np.min(m)/k,np.mean(m)/k,np.std(m)/k]
        else:
            return [len(data)/k,len(m)/k,-99,-99,-99,-99]
    else:
        m=list(set(data[col]))
        return [len(data)/k,len(m)/k,np.max(m)/k,np.min(m)/k,np.mean(m)/k,np.std(m)/k]
            
def create_set(user_list,data,cols,flag):
    '''
    定义构造数据集函数
    user_list ---list用户ID列表
    data  ---DataFrame数据
    flag ---str标识符
    return ---DataFrame特征数据集
    '''
    
    if flag=='f6':  #alldata：所有数据表
        m=[] #用于存放所有用户的特征
        for user in user_list:  #遍历每个用户，构造其特征
            new_data=[] #存放该用户的特征
            user_data=data[data['企业名称']==user].fillna(-99).reset_index(drop=True) #返回该用户的数据
            for col in cols:
                if col not in ['企业名称']:
                    new_data=new_data+create_feat(col+'_num',user_data) #字段col已经变成col+'_num'
            
            m.append([user]+new_data)
        return pd.DataFrame(m,columns=['UID']+['all{}'.format(i) for i in range(len(new_data))])
cols=feat6.columns
for col in cols:
    if col not in ['企业名称']:
        new_feat=count_num(col,feat6)
        feat6=pd.merge(feat6,new_feat,how='left',on=[col]).fillna(-99)
        #alldata.drop([col],axis=1,inplace=True)

all_user_list=list(set(feat6['企业名称'])) #用户列表
if not os.path.exists('feat6_16.csv'):
    feat6_16=create_set(all_user_list,feat6,cols=cols,flag='f6')
    feat6_16.to_csv('feat6_16.csv',index=None,encoding='utf8')
else:
    feat6_16=pd.read_csv('feat6_16.csv',encoding='utf8')
    feat6_16.columns=['企业名称']+list(feat6_16.columns[1:])
del feat6

#----------------------------feat3特征提取------
#任务编号
col='任务编号'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat3_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
del per
#统一社会信用代码
col='统一社会信用代码'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat3_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
del per
#组织机构代码
col='组织机构代码'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat3_3=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
del per
#行政区划
col='行政区划'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat3_4.csv'):
    feat3_4=count_col(per,col).fillna(0)
    feat3_4.to_csv('feat3_4.csv',index=None,encoding='utf8')
else:
    feat3_4=pd.read_csv('feat3_4.csv',encoding='utf8')
del per
#经济类型
col='经济类型'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat3_5.csv'):
    feat3_5=count_col(per,col).fillna(0)
    feat3_5.to_csv('feat3_5.csv',index=None,encoding='utf8')
else:
    feat3_5=pd.read_csv('feat3_5.csv',encoding='utf8')
del per
#经济类型
col='企业类型代码'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat3_6.csv'):
    feat3_6=count_col(per,col).fillna(0)
    feat3_6.to_csv('feat3_6.csv',index=None,encoding='utf8')
else:
    feat3_6=pd.read_csv('feat3_6.csv',encoding='utf8')
del per
#任务编号:
col='任务编号'
per=feat3[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat3_7.csv'):
    feat3_7=count_col(per,col).fillna(0)
    feat3_7.to_csv('feat3_7.csv',index=None,encoding='utf8')
else:
    feat3_7=pd.read_csv('feat3_7.csv',encoding='utf8')
del per

#feat3编码：
feat3_label_code=label_code(feat3)
feat3=feat3_label_code.copy()
cols=feat3.columns
for col in cols:
    if col not in ['企业名称']:
        new_feat=count_num(col,feat3)
        feat3=pd.merge(feat3,new_feat,how='left',on=[col]).fillna(-99)
        #alldata.drop([col],axis=1,inplace=True)

all_user_list=list(set(feat3['企业名称'])) #用户列表
if not os.path.exists('feat3_8.csv'):
    feat3_8=create_set(all_user_list,feat3,cols=cols,flag='f6')
    feat3_8.columns=['企业名称']+list(feat3_8.columns[1:])
    feat3_8.to_csv('feat3_8.csv',index=None,encoding='utf8')
else:
    feat3_8=pd.read_csv('feat3_8.csv',encoding='utf8')
    feat3_8.columns=['企业名称']+list(feat3_8.columns[1:])
del feat3
#------------------------feat5特征提取---------------------------------------
#登记注册类型
col='登记注册类型'
per=feat5[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat5_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#组织机构代码
col='组织机构代码'
per=feat5[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat5_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
del per
#feat5编码：
feat5=label_code(feat5)
#------------------------feat9特征提取---------------------------------------
#关联机构设立登记表主键ID
col='关联机构设立登记表主键ID'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
        
#信息提供部门编码
col='信息提供部门编码'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#行政相对人代码_1
col='行政相对人代码_1'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_3=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#项目名称
col='项目名称'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_4=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#审批类别
col='审批类别'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_5=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#许可机关
col='许可机关'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_6=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#地方编码
col='地方编码'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_7=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#数据状态_1
col='数据状态_1'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat9_8.csv'):
    feat9_8=count_col(per,col).fillna(0)
    feat9_8.to_csv('feat9_8.csv',index=None,encoding='utf8')
else:
    feat9_8=pd.read_csv('feat9_8.csv',encoding='utf8')
#数据状态_2
col='数据状态_2'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat9_9.csv'):
    feat9_9=count_col(per,col).fillna(0)
    feat9_9.to_csv('feat9_9.csv',index=None,encoding='utf8')
else:
    feat9_9=pd.read_csv('feat9_9.csv',encoding='utf8')
del per
#任务编号
col='任务编号'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_10=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#备注
col='备注'
per=feat9[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat9_11=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
del feat9
#-----------------------feat1特征提取--------------------------------------
#feat1编码：
feat1=label_code(feat1)
#-----------------------feat8特征提取--------------------------------------
#备注
col='关联机构设立登记表主键ID'
per=feat8[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat8_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#备注
col='公布方式及网址'
per=feat8[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat8_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#feat8编码：
feat8=label_code(feat8)
#-----------------------feat12特征提取--------------------------------------
#备注
col='关联机构设立登记表主键ID'
per=feat12[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat12_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数

col='信息提供部门编码'
per=feat12[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat12_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='认定机关全称'
per=feat12[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat12_3=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='数据状态'
per=feat12[['企业名称',col,'数据来源',\
            '信息提供部门编码','任务编号',\
            '组织机构代码','工商注册号',\
            '统一社会信用代码']].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat12_4=per
del feat12
#-----------------------feat7特征提取--------------------------------------
#备注
col='组织机构代码'
per=feat7[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat7_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='登记注册类型'
per=feat7[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat7_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#feat7编码：
feat7=label_code(feat7)
#-----------------------feat2特征提取--------------------------------------
#分支机构省份
col='分支机构省份'
per=feat2[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat2_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
if not os.path.exists('feat2_2.csv'):
    feat2_2=count_col(per,col).fillna(0)
    feat2_2.to_csv('feat2_2.csv',index=None,encoding='utf8')
else:
    feat2_2=pd.read_csv('feat2_2.csv',encoding='utf8')
#分支机构省份
col='分支行业门类'
per=feat2[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat2_3=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
if not os.path.exists('feat2_4.csv'):
    feat2_4=count_col(per,col).fillna(0)
    feat2_4.to_csv('feat2_4.csv',index=None,encoding='utf8')
else:
    feat2_4=pd.read_csv('feat2_4.csv',encoding='utf8')
#分支机构类型
col='分支机构类型'
per=feat2[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat2_5=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
if not os.path.exists('feat2_6.csv'):
    feat2_6=count_col(per,col).fillna(0)
    feat2_6.to_csv('feat2_6.csv',index=None,encoding='utf8')
else:
    feat2_6=pd.read_csv('feat2_6.csv',encoding='utf8')
#分支机构类型
col='分支机构状态'
per=feat2[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat2_7=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#分支机构区县
col='分支机构区县'
per=feat2[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat2_8=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#分支机构区县
col='分支行业代码'
per=feat2[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat2_9=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
del feat2
#-----------------------feat10特征提取--------------------------------------
#分支机构区县
col='关联机构设立登记表主键ID'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat10_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#分支机构区县
col='组织机构代码'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat10_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#分支机构区县
col='工商注册号'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat10_3=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='统一社会信用代码'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat10_4=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='证书编号'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat10_5=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='年检机关全称'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_6.csv'):
    feat10_6=count_col(per,col).fillna(0)
    feat10_6.to_csv('feat10_6.csv',index=None,encoding='utf8')
else:
    feat10_6=pd.read_csv('feat10_6.csv',encoding='utf8')
col='年检事项名称'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_7.csv'):
    feat10_7=count_col(per,col).fillna(0)
    feat10_7.to_csv('feat10_7.csv',index=None,encoding='utf8')
else:
    feat10_7=pd.read_csv('feat10_7.csv',encoding='utf8')
col='信息提供部门编码'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_8.csv'):
    feat10_8=count_col(per,col).fillna(0)
    feat10_8.to_csv('feat10_8.csv',index=None,encoding='utf8')
else:
    feat10_8=pd.read_csv('feat10_8.csv',encoding='utf8')
col='年检结果'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_9.csv'):
    feat10_9=count_col(per,col).fillna(0)
    feat10_9.to_csv('feat10_9.csv',index=None,encoding='utf8')
else:
    feat10_9=pd.read_csv('feat10_9.csv',encoding='utf8')
col='数据状态'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_10.csv'):
    feat10_10=count_col(per,col).fillna(0)
    feat10_10.to_csv('feat10_10.csv',index=None,encoding='utf8')
else:
    feat10_10=pd.read_csv('feat10_10.csv',encoding='utf8')
col='数据来源'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_11.csv'):
    feat10_11=count_col(per,col).fillna(0)
    feat10_11.to_csv('feat10_11.csv',index=None,encoding='utf8')
else:
    feat10_11=pd.read_csv('feat10_11.csv',encoding='utf8')
col='权力编码'
per=feat10[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
if not os.path.exists('feat10_12.csv'):
    feat10_12=count_col(per,col).fillna(0)
    feat10_12.to_csv('feat10_12.csv',index=None,encoding='utf8')
else:
    feat10_12=pd.read_csv('feat10_12.csv',encoding='utf8')
#-----------------------feat11特征提取--------------------------------------
#分支机构区县
col='工作经验'
per=feat11[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat11_1=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#分支机构区县
col='工作地点'
per=feat11[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat11_2=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#分支机构区县
col='招聘人数'
per=feat11[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat11_3=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='职位月薪'
per=feat11[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat11_4=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='最低学历'
per=feat11[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat11_5=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
col='业务主键'
per=feat11[['企业名称',col]].drop_duplicates().reset_index(drop=True)
print(len(per),len(set(per[col])))
per=per[per[col]!=-99].reset_index(drop=True)
print(len(per),len(set(per[col])))
feat11_6=per.groupby(['企业名称'],as_index=False)[col].agg({
        col+'_count':'count'}) #频数
#-------------------------结合特征---------------------------------
print('结合feat6特征...')
train=pd.merge(train_id,feat6_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_4,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_5,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_6,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_7,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_8,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_9,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_10,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_11,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_12,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_13,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_14,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_15,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat6_16,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test_id,feat6_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_4,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_5,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_6,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_7,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_8,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_9,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_10,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_11,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_12,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_13,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_14,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_15,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat6_16,how='left',on=['企业名称']).fillna(-99)
print('结合feat6特征完毕')
print('结合feat3特征...')
train=pd.merge(train,feat3_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_4,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_5,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_6,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_7,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_8,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat3_label_code,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat3_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_4,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_5,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_6,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_7,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_8,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat3_label_code,how='left',on=['企业名称']).fillna(-99)
print('结合feat3特征完毕')

print('结合feat5特征...')
train=pd.merge(train,feat5_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat5_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat5,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat5_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat5_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat5,how='left',on=['企业名称']).fillna(-99)
print('结合feat5特征完毕')
print('结合feat9特征...')
train=pd.merge(train,feat9_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_4,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_5,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_6,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_7,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_8,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_9,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_10,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat9_11,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat9_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_4,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_5,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_6,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_7,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_8,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_9,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_10,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat9_11,how='left',on=['企业名称']).fillna(-99)
print('结合feat9特征完毕')

print('结合feat1特征...')
train=pd.merge(train,feat1,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat1,how='left',on=['企业名称']).fillna(-99)
print('结合feat1特征完毕')

print('结合feat8特征...')
train=pd.merge(train,feat8_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat8_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat8,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat8_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat8_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat8,how='left',on=['企业名称']).fillna(-99)
print('结合feat8特征完毕')

print('结合feat12特征...')
train=pd.merge(train,feat12_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat12_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat12_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat12_4,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat12_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat12_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat12_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat12_4,how='left',on=['企业名称']).fillna(-99)
print('结合feat12特征完毕')

print('结合feat7特征...')
train=pd.merge(train,feat7_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat7_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat7,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat7_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat7_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat7,how='left',on=['企业名称']).fillna(-99)
print('结合feat7特征完毕')

print('结合feat2特征...')
train=pd.merge(train,feat2_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_4,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_5,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_6,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_7,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_8,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat2_9,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat2_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_4,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_5,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_6,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_7,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_8,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat2_9,how='left',on=['企业名称']).fillna(-99)
print('结合feat2特征完毕')

print('结合feat10特征...')
train=pd.merge(train,feat10_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_4,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_5,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_6,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_7,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_8,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_9,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_10,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_11,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat10_12,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat10_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_4,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_5,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_6,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_7,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_8,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_9,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_10,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_11,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat10_12,how='left',on=['企业名称']).fillna(-99)
print('结合feat10特征完毕')

print('结合feat11特征...')
train=pd.merge(train,feat11_1,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat11_2,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat11_3,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat11_4,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat11_5,how='left',on=['企业名称']).fillna(-99)
train=pd.merge(train,feat11_6,how='left',on=['企业名称']).fillna(-99)

test=pd.merge(test,feat11_1,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat11_2,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat11_3,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat11_4,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat11_5,how='left',on=['企业名称']).fillna(-99)
test=pd.merge(test,feat11_6,how='left',on=['企业名称']).fillna(-99)
print('结合feat11特征完毕')

print('构造count特征...')
alldata1_len=alldata1.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata1': 'count'})
alldata2_len=alldata2.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata2': 'count'})
alldata3_len=alldata3.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata3': 'count'})
alldata4_len=alldata4.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata4': 'count'})
alldata5_len=alldata5.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata5': 'count'})
alldata6_len=alldata6.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata6': 'count'})
alldata7_len=alldata7.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata7': 'count'})
alldata8_len=alldata8.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata8': 'count'})
alldata9_len=alldata9.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata9': 'count'})
alldata10_len=alldata10.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata10': 'count'})
alldata11_len=alldata11.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata11': 'count'})
alldata12_len=alldata12.groupby(['企业名称'], as_index=False)['企业名称'].agg({'alldata12': 'count'})
del alldata1,alldata2,alldata3,alldata4,alldata5,alldata6,alldata7,\
alldata8,alldata9,alldata10,alldata11,alldata12
print('构造count特征完毕！')

print('结合count特征...')
train=pd.merge(train,alldata1_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata2_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata3_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata4_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata5_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata6_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata7_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata8_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata9_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata10_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata11_len,how='left',on=['企业名称']).fillna(0)
train=pd.merge(train,alldata12_len,how='left',on=['企业名称']).fillna(0)

test=pd.merge(test,alldata1_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata2_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata3_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata4_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata5_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata6_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata7_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata8_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata9_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata10_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata11_len,how='left',on=['企业名称']).fillna(0)
test=pd.merge(test,alldata12_len,how='left',on=['企业名称']).fillna(0)
print('结合count特征完毕！')

print('---------------------所有特征工程结束！------------------------')
print(train.shape,test.shape)

#--------------------两个训练标签--------------------
train_y1=train['label1']
train_y2=train['label2']
train.to_csv('train.csv',index=None,encoding='utf8')
test.to_csv('test.csv',index=None,encoding='utf8')

train= pd.read_csv('train.csv',encoding='utf8')
test= pd.read_csv('test.csv',encoding='utf8')

train_id=train[['企业名称']]
train=train.drop(['企业名称','label1','label2'],axis=1) #训练数据
submit=pd.DataFrame(list(set(test['企业名称'])),columns=['企业名称']) #待提交企业名称
submit1=test[['企业名称']] #待预测企业名称
test=test.drop(['企业名称'],axis=1) #测试数据

#test=test[train.columns]
train=np.array(train) #数据类型转换
test=np.array(test) #数据类型转换
#np.save('train.npy',train)
#np.save('test.npy',test)

##特征选择:
print('特征选择中...')
'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
sm = SelectFromModel(GradientBoostingClassifier(random_state=1))
train1= sm.fit_transform(train, train_y1)
test1= sm.transform(test)
train2= sm.fit_transform(train, train_y2)
test2= sm.transform(test)
'''
print('特征选择完毕！')

'''
##--------------------------xgb-------------------------------------------------
n_folds=10
kf = KFold(n_splits =n_folds, shuffle=True, random_state=42)
stack_train = np.zeros((len(train_y1), 1))
stack_test = np.zeros((len(test), 1))
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print('第一个模型训练...\n第%d/%dflod训练中...'%(i+1,n_folds))
    x_tr = train[train_index] #训练集
    y_tr = train_y1[train_index]
    x_te = train[test_index] #验证集
    y_te=train_y1[test_index]
    model=xgb_model(x_tr, y_tr,x_te,y_te)
    score_va =model.predict(xgb.DMatrix(x_te),ntree_limit=model.best_ntree_limit)
    score_te =model.predict(xgb.DMatrix(test),ntree_limit=model.best_ntree_limit)
    stack_train[test_index,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
train_id['y1']=stack_train
submit1['y1']=stack_test

n_folds=10
kf = KFold(n_splits =n_folds, shuffle=True, random_state=42)
stack_train = np.zeros((len(train_y2), 1))
stack_test = np.zeros((len(test), 1))
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print('第二个模型训练...\n第%d/%dflod训练中...'%(i+1,n_folds))
    x_tr = train[train_index] #训练集
    y_tr = train_y2[train_index]
    x_te = train[test_index] #验证集
    y_te=train_y2[test_index]
    model=xgb_model(x_tr, y_tr,x_te,y_te)
    score_va =model.predict(xgb.DMatrix(x_te),ntree_limit=model.best_ntree_limit)
    score_te =model.predict(xgb.DMatrix(test),ntree_limit=model.best_ntree_limit)
    stack_train[test_index,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
train_id['y2']=stack_train
submit1['y2']=stack_test
train_id.to_csv('train_id_xgb5cv_1.csv',index=None,encoding='utf8')
submit1.to_csv('submit1_xgb5cv_1.csv',index=None,encoding='utf8')

#保存结果 0.9373
submit=pd.merge(submit,submit1.groupby(['企业名称'],\
                        as_index=False)['y1'].agg({'y1':'mean'}),\
                        how='left',on=['企业名称'])
submit=pd.merge(submit,submit1.groupby(['企业名称'],\
                        as_index=False)['y2'].agg({'y2':'mean'}),\
                        how='left',on=['企业名称'])
submit.columns=['EID','FORTARGET1','FORTARGET2']
submit.to_csv('compliance_assessment.csv',header=None,index=None)
'''


##--------------------------lgb-----------------------------------------------
import lightgbm  as lgb
n_folds=10
kf = KFold(n_splits =n_folds, shuffle=True, random_state=42)
stack_train = np.zeros((len(train_y1), 1))
stack_test = np.zeros((len(test), 1))
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print('第一个模型训练...\n第%d/%dflod训练中...'%(i+1,n_folds))
    x_tr = train[train_index] #训练集
    y_tr = train_y1[train_index]
    x_te = train[test_index] #验证集
    y_te=train_y1[test_index]
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 2 ** 4,
    'learning_rate': 0.03,#0.03
    'verbose': 1,
    'seed': 2018,
    }
    lgb_train = lgb.Dataset(x_tr, y_tr)
    lgb_eval = lgb.Dataset(x_te,y_te, reference=lgb_train)
    model = lgb.train(params,
                lgb_train,
                num_boost_round=8000,
                valid_sets=lgb_eval,
                verbose_eval=100,
                early_stopping_rounds=100
                )
    score_va =model.predict(x_te, num_iteration=model.best_iteration)
    score_te =model.predict(test, num_iteration=model.best_iteration)
    stack_train[test_index,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
train_id['y1']=stack_train
submit1['y1']=stack_test

n_folds=10
kf = KFold(n_splits =n_folds, shuffle=True, random_state=42)
stack_train = np.zeros((len(train_y2), 1))
stack_test = np.zeros((len(test), 1))
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print('第一个模型训练...\n第%d/%dflod训练中...'%(i+1,n_folds))
    x_tr = train[train_index] #训练集
    y_tr = train_y2[train_index]
    x_te = train[test_index] #验证集
    y_te=train_y2[test_index]
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 2 ** 5,
    'learning_rate': 0.1,
    'verbose': 1,
    'seed': 2018,
    }
    lgb_train = lgb.Dataset(x_tr, y_tr)
    lgb_eval = lgb.Dataset(x_te,y_te, reference=lgb_train)
    model = lgb.train(params,
                lgb_train,
                num_boost_round=8000,
                valid_sets=lgb_eval,
                verbose_eval=100,
                early_stopping_rounds=100
                )
    score_va =model.predict(x_te, num_iteration=model.best_iteration)
    score_te =model.predict(test, num_iteration=model.best_iteration)
    stack_train[test_index,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
train_id['y2']=stack_train
submit1['y2']=stack_test
train_id.to_csv('train_id_lgb5cv_1.csv',index=None,encoding='utf8')
submit1.to_csv('submit1_lgb5cv_1.csv',index=None,encoding='utf8')

#保存结果 0.9378
submit=pd.merge(submit,submit1.groupby(['企业名称'],\
                        as_index=False)['y1'].agg({'y1':'mean'}),\
                        how='left',on=['企业名称'])
submit=pd.merge(submit,submit1.groupby(['企业名称'],\
                        as_index=False)['y2'].agg({'y2':'mean'}),\
                        how='left',on=['企业名称'])
submit.columns=['EID','FORTARGET1','FORTARGET2']
submit.to_csv('compliance_assessment.csv',header=None,index=None)
