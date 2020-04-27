# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:20:17 2020

@author: dshre
"""

import pandas as pd

lung_cancer=pd.read_csv('final_respiratory_cancer.csv',index_col=0)

lung_cancer.replace(to_replace={'MAR_STAT':9,'RACE1V':99,'AGE_DX':999,'SEQ_NUM':99,
                                  'Lateral':9,'GRADE':9,'DX_CONF':9,'CSEXTEN':999,
                                  'CSLYMPHN':999,'DAJCCT':88,'DAJCCN':88,'DAJCCM':88,
                                  'SURGSCOF':9,'SURGSITF':9,'NO_SURG':9,'AGE_1REC':99,
                                  'RAC_RECA':9,'RAC_RECY':9,'HST_STGA':9,'INTPRIM':9,
                                  'ERSTATUS':9,'PRSTATUS':9,'SRV_TIME_MON':9999,'SRV_TIME_MON_FLAG':9,
                                  'HER2':9,'BRST_SUB':9,'MALIGCOUNT':99,'BENBORDCOUNT':99,
                                  'RAD_SURG':9},value=pd.np.nan,inplace=True)

lung_cancer.replace({'EOD10_PN':{95:pd.np.nan,96:pd.np.nan,97:pd.np.nan,98:pd.np.nan,99:pd.np.nan},
                       'EOD10_NE':{95:pd.np.nan,96:pd.np.nan,97:pd.np.nan,98:pd.np.nan,99:pd.np.nan},
                       'CSTUMSIZ':{990:0,991:10,992:20,993:30,994:40,995:50,996:pd.np.nan,997:pd.np.nan,998:pd.np.nan,
                                   999:pd.np.nan,888:pd.np.nan},
                       'DAJCCSTG':{88:pd.np.nan,90:pd.np.nan,99:pd.np.nan},
                       'DSS1977S':{8:pd.np.nan,9:pd.np.nan},'SURGPRIF':{90:pd.np.nan,98:pd.np.nan,99:pd.np.nan},
                       'ADJTM_6VALUE':{88:pd.np.nan,99:pd.np.nan},'ADJNM_6VALUE':{88:pd.np.nan,99:pd.np.nan},
                       'ADJM_6VALUE':{88:pd.np.nan,99:pd.np.nan},'ADJAJCCSTG':{88:pd.np.nan,90:pd.np.nan,99:pd.np.nan}
                       },inplace=True)

lung_cancer.dropna(axis=0,how='any',subset=['YEAR_DX','SRV_TIME_MON'],inplace=True)

columns=lung_cancer.isna().sum(axis=0)/len(lung_cancer)

columns_list=list(columns[columns<0.2].index)

lung_cancer=lung_cancer.filter(items=columns_list,axis=1)

stats=lung_cancer.describe().loc['50%']
catg=lung_cancer.mode()

drop_cols=['MDXRECMP','YEAR_DX','CSVFIRST','CSVLATES','CSVCURRENT','ICCC3WHO',
           'ICCC3XWHO','CODPUB','CODPUBKM','STAT_REC','IHSLINK','VSRTSADX','ODTHCLASS',
           'CSTSEVAL','CSRGEVAL','CSMTEVAL','ST_CNTY','SRV_TIME_MON','SRV_TIME_MON_FLAG',
           '1year_survival','5year_survival']
catg_cols=['REG','MAR_STAT','RACE1V','SEX','PRIMSITE','LATERAL','BEHO2V', 'BEHO3V',
           'DX_CONF','REPT_SRC','CSMETSDX','DAJCCT','DAJCCN','DAJCCM','DAJCCSTG','DSS1977S',
           'SCSSM2KO','SURGPRIF','SURGSCOF','SURGSITF','NO_SURG','TYPE_FU','AGE_1REC','SITERWHO',
           'ICDOTO9V','ICDOT10V','BEHTREND','HISTREC','HISTRECB','CS0204SCHEMA','RAC_RECA',
           'RAC_RECY','ORIGRECB','HST_STGA','FIRSTPRM','SUMM2K','AYASITERWHO','LYMSUBRWHO',
           'INTPRIM','CSSCHEMA','ANNARBOR','RADIATNR','RAD_SURG','CHEMO_RX_REC']
num_cols=['AGE_DX','YR_BRTH','SEQ_NUM','EOD10_NE','CSEXTEN','CSLYMPHN','HISTO2V',
          'HISTO3V','CS1SITE','CS25SITE','REC_NO','MALIGCOUNT','BENBORDCOUNT']

values=dict()
for i in catg_cols:
    values[i]=catg[i][0]
for i in num_cols:
    values[i]=stats[i]

lung_cancer.fillna(value=values,inplace=True)

'''
lung_cancer['survival_classes']=lung_cancer.apply(lambda row: 
    '<=1yrs' if (row.SRV_TIME_MON<=60) 
    else ('5-7yrs'  if (row.SRV_TIME_MON<=84)
              else ('7-10yrs' if (row.SRV_TIME_MON<=120) else '>10yrs')),axis=1)
'''
from numpy.random import seed
from tensorflow import set_random_seed

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
#from sklearn.pipeline import Pipeline
from keras import backend
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data=lung_cancer.drop(columns=drop_cols)
target=pd.DataFrame(lung_cancer['SRV_TIME_MON'])
data=pd.get_dummies(data,prefix=catg_cols,columns=catg_cols,drop_first=False)

#scaler1=StandardScaler()
#scaler2=StandardScaler()

scaler1=MinMaxScaler()
scaler2=MinMaxScaler()

data_scaled=scaler1.fit_transform(data)
target_scaled=scaler2.fit_transform(target)

data_scaled=data_scaled.reshape(data_scaled.shape[0],1,data_scaled.shape[1])

def rmse(y_true,y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))

def r_sq(y_true, y_pred):
    SS_res =  backend.sum(backend.square( y_true-y_pred )) 
    SS_tot = backend.sum(backend.square( y_true - backend.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )

def build_model():
    model=Sequential()
    
    model.add(Conv1D(50, 1, input_shape=train_data.shape[1:], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(20, 1, input_shape=train_data.shape[1:], activation='relu')) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse',rmse,r_sq])
    
    return model

#estimators = []
#estimators.append(('standardize', MinMaxScaler()))
#estimators.append(('mlp',KerasRegressor(build_fn=build_model, epochs=20, batch_size=5, verbose=1)))
estimators=KerasRegressor(build_fn=build_model, epochs=50, batch_size=64, verbose=1)

#pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(estimators, data_scaled, target_scaled, cv=kfold)

print("MSE Score: %.6f (%.6f) MSE" % (results.mean(), results.std()))

train_data,test_data,train_target,test_target=train_test_split(data_scaled,target_scaled,test_size=0.2,random_state=21)
'''
from keras.callbacks import ModelCheckpoint

chk = ModelCheckpoint("Modelling/ibd.h5", monitor='loss', save_best_only=True, mode='min')
callback_list=[chk]
'''
#estimators.fit(train_data,train_target,callbacks=callback_list)
history=estimators.fit(train_data,train_target,validation_data=(test_data,test_target))

test_pred=pd.DataFrame(estimators.predict(test_data),index=test_data.index,columns=['Pred'])
print("MSE Cross Val Score: %.6f (%.6f) MSE" % (results.mean(), results.std()))
print('MSE=',mean_squared_error(test_target,test_pred))
print('RMSE=',math.sqrt(mean_squared_error(test_target,test_pred)))
print('R Square=',r2_score(test_target,test_pred))

epoch_num = pd.np.arange(0, 50)
plt.figure()
plt.plot(epoch_num, history.history["mean_squared_error"], label="train_mse")
plt.plot(epoch_num, history.history["val_mean_squared_error"], label="val_mse")
#plt.plot(epoch_num, history.history["r_sq"], label="train_r_square")
#plt.plot(epoch_num, history.history["val_r_sq"], label="val_r_square")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Error")
plt.legend()
plt.savefig('Lung Cancer CNN.png')

compare_test_data=pd.DataFrame({'SRV_TIME_MON':test_target[:,0],'Pred':test_pred})
compare_test_data['Months Unscaled']=scaler2.inverse_transform(test_target)

compare_test_data['survival_classes']=compare_test_data.apply(lambda row: 
    '<=6months' if (row['Months Unscaled']<=6) 
    else ('0.5-2yrs'  if (row['Months Unscaled']<=24) else '>2yrs'),axis=1)

classes=list(compare_test_data['survival_classes'].unique())
dct1=dict()

for each_class in classes:
    subset=compare_test_data[compare_test_data['survival_classes']==each_class]
    mse=mean_squared_error(subset['SRV_TIME_MON'],subset['Pred'])
    rootmse=math.sqrt(mean_squared_error(subset['SRV_TIME_MON'],subset['Pred']))
    dct1[each_class]=[mse,rootmse,len(subset)]

months=list(compare_test_data['Months Unscaled'].unique())
dct2=dict()

for month in months:
    subset=compare_test_data[compare_test_data['Months Unscaled']==month]
    mse=mean_squared_error(subset['SRV_TIME_MON'],subset['Pred'])
    rootmse=math.sqrt(mean_squared_error(subset['SRV_TIME_MON'],subset['Pred']))
    dct2[month]=[mse,rootmse,len(subset)]

class_level=pd.DataFrame.from_dict(dct1,orient='index',columns=['MSE','RMSE','Support Count'])
month_level=pd.DataFrame.from_dict(dct2,orient='index',columns=['MSE','RMSE','Support Count'])
month_level.sort_index(axis=0,inplace=True)

plt.clf()
plt.figure()
plt.title('No. of Survival Months vs MSE')
ax1=month_level['Support Count'].plot(use_index=True,kind='bar',color='b',xticks=range(0,160,10),legend=True)
ax2=month_level['MSE'].plot(kind='line',color='k',secondary_y=True,legend=True,mark_right=False)
ax1.set_xlabel('No. of Survival Months')
ax1.set_ylabel('Support Count')
ax2.set_ylabel('MSE')
plt.savefig('Lung Cancer CNN MSE.png')

plt.clf()
plt.figure()
plt.title('No. of Survival Months vs RMSE')
ax1=month_level['Support Count'].plot(use_index=True,kind='bar',color='b',xticks=range(0,160,10),legend=True)
ax2=month_level['RMSE'].plot(kind='line',color='r',secondary_y=True,legend=True,mark_right=False)
ax1.set_xlabel('No. of Survival Months')
ax1.set_ylabel('Support Count')
ax2.set_ylabel('RMSE')
plt.savefig('Lung Cancer CNN RMSE.png')

