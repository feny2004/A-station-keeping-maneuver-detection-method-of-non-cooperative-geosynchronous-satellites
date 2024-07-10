# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 09:42:27 2021
1D-CNN及其余对比算法
所有的都在里面，包括指标计算。
计算roc-auc时应注意，输入标签有的需要是编码过的，有的方法则要求是直接的0 1 2此处应注意
@author: Lenovo
"""


import numpy as np
# import pandas as pd 

# import keras
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout

from  keras.layers import Conv1D,MaxPooling1D
from  keras.layers import Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from  keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from  keras import utils as np_utils
import sklearn

from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score



trainx = np.load('trainx.npy')
testx = np.load('testx.npy')
trainy = np.load('trainy.npy')
testy = np.load('testy.npy')

trainx=np.expand_dims(trainx,axis=3)
testx=np.expand_dims(testx,axis=3)
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(trainy)
label_encode=encoder.fit_transform(testy)
trainy= np_utils.to_categorical(Y_encoded)
testy= np_utils.to_categorical(label_encode)



trainX,trainy=sklearn.utils.shuffle(trainx,trainy,random_state=1337)
testX,testy=sklearn.utils.shuffle(testx,testy,random_state=1337)

#将数据转成一维的形式，拉平即可，原来是（20，6），现在即为20*6=120
trainX = np.squeeze(trainX, axis=None)
testX = np.squeeze(testX, axis=None)

#以下两行代码1dcnn不需要,其他对比算法需要
trainX=trainX.reshape(-1,120)
testX=testX.reshape(-1,120)

'''
模型svm
'''
'''
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

model = OneVsRestClassifier(svm.LinearSVC(random_state = 0, max_iter=10000,verbose = 1))

model.fit(trainX, trainy)

y_score = model.decision_function(testX) 

y_pred=model.predict(testX)

precision_score=precision_score(testy, y_pred,average='macro')
recall_score=recall_score(testy, y_pred,average='macro')
macro_f1=f1_score(testy, y_pred,average='macro')
score=model.score(testX,testy)

roc_auc_score=roc_auc_score(testy, y_score,average='macro', multi_class='ovo')
ff=f1_score(testy,y_pred,average='micro')
'''

'''
Lighrlgb模型，结果比2DCNN好
'''
'''
import lightgbm as lgb
trainy = np.asarray(trainy,'int64')
de_onehot = []
for i in range(len(trainy)):
    if trainy[i][0] == 1:
        de_onehot.append(0)
    elif trainy[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

print("----------after the de-one-hot encoding----------")
de_onehot = np.array(de_onehot)
de_onehot_train = de_onehot.reshape(-1,1)

testy = np.asarray(testy,'int64')
de_onehot = []
for i in range(len(testy)):
    if testy[i][0] == 1:
        de_onehot.append(0)
    elif testy[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

print("----------after the de-one-hot encoding----------")
de_onehot_testy = np.array(de_onehot)
de_onehot_test = de_onehot_testy.reshape(-1,1)

# 训练

train_data=lgb.Dataset(trainX,label=de_onehot_train)
validation_data=lgb.Dataset(testX,label=de_onehot_test)
params={
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':6,
    'objective':'multiclass',
    'num_class':4,  
}
clf=lgb.train(params,train_data,valid_sets=[validation_data])


# 1、AUC
y_pred_pa = clf.predict(testX)  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
y_pred = y_pred_pa.argmax(axis=1)
#  3、经典-精确率、召回率、F1分数
precision_score=precision_score(de_onehot_test, y_pred,average='macro')
recall_score=recall_score(de_onehot_test, y_pred,average='macro')
# f1_score=f1_score(de_onehot_test, y_pred,average='macro')
macro_f1=f1_score(de_onehot_test, y_pred,average='macro')
ff=f1_score(de_onehot_test,y_pred,average='micro')
'''

'''
对比模型线性逻辑回归
'''
'''
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=0, solver='sag',multi_class='ovr', verbose = 1)

trainy = np.asarray(trainy,'int64')
de_onehot = []
for i in range(len(trainy)):
    if trainy[i][0] == 1:
        de_onehot.append(0)
    elif trainy[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

de_onehot = np.array(de_onehot)
de_onehot_train = de_onehot.reshape(-1,1)

testy = np.asarray(testy,'int64')
de_onehot = []
for i in range(len(testy)):
    if testy[i][0] == 1:
        de_onehot.append(0)
    elif testy[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

print("----------after the de-one-hot encoding----------")
de_onehot_testy = np.array(de_onehot)
de_onehot_test = de_onehot_testy.reshape(-1,1)

lr_clf.fit(trainX, de_onehot_train)

y_pred_pa = lr_clf.predict_proba(testX)
y_pred_l = lr_clf.predict(testX)

precision_score=precision_score(de_onehot_test, y_pred_l,average='macro')
recall_score=recall_score(de_onehot_test, y_pred_l,average='macro')
macro_f1=f1_score(de_onehot_test, y_pred_l,average='macro')
micro_f1=f1_score(de_onehot_test, y_pred_l,average='micro')
scores = lr_clf.score(testX, de_onehot_test)
roc_auc_score=roc_auc_score(testy, y_pred_pa,average='macro', multi_class='ovo')
'''

'''
对比模型结果
'''

'''
对比模型DT
'''
'''
from sklearn.tree import DecisionTreeClassifier

trainy = np.asarray(trainy,'int64')
de_onehot = []
for i in range(len(trainy)):
    if trainy[i][0] == 1:
        de_onehot.append(0)
    elif trainy[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

de_onehot = np.array(de_onehot)
de_onehot_train = de_onehot.reshape(-1,1)

# testy = np.asarray(testy,'int64')
de_onehot_testy = []
for i in range(len(testy)):
    if testy[i][0] == 1:
        de_onehot_testy.append(0)
    elif testy[i][1] == 1:
        de_onehot_testy.append(1)
    else:
        de_onehot_testy.append(2)

print("----------after the de-one-hot encoding----------")
de_onehot_testy = np.array(de_onehot_testy)
de_onehot_test = de_onehot_testy.reshape(-1,1)



model_dt = DecisionTreeClassifier(criterion='entropy'
                                  ,min_samples_leaf=3,random_state=24).fit(trainX,de_onehot_train)  # 

y_pred = model_dt.predict_proba(testX)

y_pred_label = model_dt.predict(testX)

score_model_dt=model_dt.score(testX,de_onehot_test)

recall_score=recall_score(de_onehot_test, y_pred_label,average='macro')
macro_f1=f1_score(de_onehot_test, y_pred_label,average='macro')
micro_f1=f1_score(de_onehot_test, y_pred_label,average='micro')
precision_score=precision_score(de_onehot_test, y_pred_label,average='macro')
roc_auc_score=roc_auc_score(de_onehot_testy, y_pred,average='macro', multi_class='ovo')
'''

'''
对比模型RF
'''
'''
from sklearn.ensemble import RandomForestClassifier
trainy = np.asarray(trainy,'int64')
de_onehot = []
for i in range(len(trainy)):
    if trainy[i][0] == 1:
        de_onehot.append(0)
    elif trainy[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

de_onehot = np.array(de_onehot)
de_onehot_train = de_onehot.reshape(-1,1)

testy = np.asarray(testy,'int64')
de_onehot_testy = []
for i in range(len(testy)):
    if testy[i][0] == 1:
        de_onehot_testy.append(0)
    elif testy[i][1] == 1:
        de_onehot_testy.append(1)
    else:
        de_onehot_testy.append(2)

de_onehot_testy = np.array(de_onehot_testy)
de_onehot_test = de_onehot_testy.reshape(-1,1)

model_rfc = RandomForestClassifier(random_state=0).fit(trainX,de_onehot_train)

y_pred = model_rfc.predict_proba(testX)

y_pred_label = model_rfc.predict(testX)

score_model_rfc=model_rfc.score(testX,de_onehot_test)

recall_score=recall_score(de_onehot_test, y_pred_label,average='macro')
macro_f1=f1_score(de_onehot_test, y_pred_label,average='macro')
micro_f1=f1_score(de_onehot_test, y_pred_label,average='micro')
precision_score=precision_score(de_onehot_test, y_pred_label,average='macro')
roc_auc_score=roc_auc_score(de_onehot_testy, y_pred,average='macro', multi_class='ovo')
'''

'''
构建模型1dcnn
'''

def baseline_model():
    model = Sequential()
    model.add(Conv1D(256,6,input_shape=(20,6), activation='tanh'))
    # model.add(Activation('tanh'))
    model.add(Conv1D(128,4, activation='tanh'))
    # model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=(2)))
    # model.add(Conv2D(64,(2,1),activation='tanh'))
    # model.add(Conv2D(64,(2,1),activation='tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(MaxPooling2D(3,3))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3,2), padding='same',activation='tanh'))
    # model.add(Conv2D(64, (3,2), padding='same',activation='tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv1D(64, 3, activation='tanh'))
    # model.add(Conv1D(64, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(192, activation='softmax'))
    model.add(Dense(96, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    print(model.summary()) # 显示网络结构
    
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])#binary_crossentropycategorical_crossentropy
    return model

# 训练分类器
estimator = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=80, verbose=1)
estimator.fit(trainX, trainy)

score = estimator.score(testX, testy,  verbose=1)
predicted =estimator.predict_proba(testX)
predicted_label = estimator.predict(testX)
# print("The accuracy of the classification model:")


y_true=testy
y_pred=predicted_label

y_true= np.asarray(y_true,'int64')
de_onehot = []
for i in range(len(y_true)):
    if y_true[i][0] == 1:
        de_onehot.append(0)
    elif y_true[i][1] == 1:
        de_onehot.append(1)
    else:
        de_onehot.append(2)

print("----------after the de-one-hot encoding----------")
de_onehotl = np.array(de_onehot)
de_onehot = de_onehotl.reshape(-1,1)

recall_score=recall_score(de_onehot, y_pred,average='macro')
Macro_f1=f1_score(de_onehot, y_pred,average='macro')
precision_score=precision_score(de_onehot, y_pred,average='macro')

roc_auc_score=roc_auc_score(de_onehotl, predicted,average='macro', multi_class='ovr')
# from sklearn.metrics import make_scorer
# param_grid = [
#     {'polynomialfeatures__degree': np.arange(2, 10).tolist(), 'logisticregression__penalty': ['l1'], 'logisticregression__C': np.arange(0.1, 2, 0.1).tolist(), 'logisticregression__solver': ['saga']}, 
#     {'polynomialfeatures__degree': np.arange(2, 10).tolist(), 'logisticregression__penalty': ['l2'], 'logisticregression__C': np.arange(0.1, 2, 0.1).tolist(), 'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']},
#     {'polynomialfeatures__degree': np.arange(2, 10).tolist(), 'logisticregression__penalty': ['elasticnet'], 'logisticregression__C': np.arange(0.1, 2, 0.1).tolist(), 'logisticregression__l1_ratio': np.arange(0.1, 1, 0.1).tolist(), 'logisticregression__solver': ['saga']}
# ]
# acc = make_scorer(roc_auc_score)
# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# search = GridSearchCV(estimator=clf,param_grid=param_grid_simple,scoring = scoring,
#                       refit='accuracy')

ff=f1_score(de_onehot,y_pred,average='micro')

# print('%s: %.2f%%' % (estimator.metrics_names[1], scores[1] * 100))
# 输出预测类别

                                    
# np.savetxt(r"E:\GEO-2\run-1\predicted.csv", predicted, fmt="%d",delimiter=",")

# 将模型转换为json并保持
model_json = estimator.model.to_json()
with open(r"E:\GEO-3\run-1d\model-1d.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
estimator.model.save_weights('model-1d.h5')
  # from sklearn.metrics import precision_score, accuracy_score,recall_score, f1_score,roc_auc_score, precision_recall_fscore_support, roc_curve, classification_report


# 加载模型用做预测
json_file = open(r"E:\GEO-3\run-1d\model-1d.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model-1d.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(testX, testy, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# 输出预测类别
predicted_1 = loaded_model.predict(testX)
predicted_label = loaded_model.predict_classes(testX)
# print("predicted label:\n " + str(predicted_label))

# plot_confuse(estimator.model, testX, testy)


# c=confusion_matrix(de_onehot, predicted_label)