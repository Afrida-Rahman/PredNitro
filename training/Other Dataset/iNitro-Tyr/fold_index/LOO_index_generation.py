from sklearn.model_selection import RepeatedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut

y_ace = []
for i in range(1044):
    y_ace.append(1);
for i in range(7669):
    y_ace.append(-1);

index_path = r'F:\recent_research_topics\nitrotyrosine\new-nitrotyrosine\other predictors\iNitro-tyr\training\fold_index'

y_ace = np.array(y_ace, dtype=np.int64)
y_train_ace = y_ace

loo = LeaveOneOut()
train_index=[]
test_index=[]
train_df=pd.DataFrame()   
test_df=pd.DataFrame()

i=0
for train_i, test_i in loo.split(y_train_ace):
    train_df1=pd.DataFrame(train_i)
    test_df1=pd.DataFrame(test_i)
    
    train_df=pd.concat([train_df,train_df1],axis=1, ignore_index=True)
    test_df=pd.concat([test_df,test_df1],axis=1, ignore_index=True)
    
    
train_df.to_csv(index_path+'\\train_index_LOO.csv', index=None, header=None)
test_df.to_csv(index_path+'\\test_index_LOO.csv', index=None, header=None)