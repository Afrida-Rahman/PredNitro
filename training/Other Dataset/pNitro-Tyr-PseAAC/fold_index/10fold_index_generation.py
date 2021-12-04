from sklearn.model_selection import RepeatedKFold
import numpy as np
import pandas as pd

y = []
for i in range(1151):
    y.append(1);
for i in range(7666):
    y.append(-1);

y = np.array(y, dtype=np.int64)
y_train = y

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
train_index=[]
test_index=[]
train_df=pd.DataFrame()   
test_df=pd.DataFrame()

i=0
for train_i, test_i in rkf.split(y_train):
    #train_index.append(train_i), test_index.append(test_i) 
    train_df1=pd.DataFrame(train_i)
    test_df1=pd.DataFrame(test_i)
    
    train_df=pd.concat([train_df,train_df1],axis=1, ignore_index=True)
    test_df=pd.concat([test_df,test_df1],axis=1, ignore_index=True)
    
    
train_df.to_csv('train_index_10fold.csv', index=None, header=None)
test_df.to_csv('test_index_10fold.csv', index=None, header=None)