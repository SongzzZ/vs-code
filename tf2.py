#!/usr/bin/env python
# coding=UTF-8
'''
@Description: tensorflow test
@Author: ZhaoSong
@LastEditors: ZhaoSong
@Date: 2019-03-05 13:19:08
@LastEditTime: 2019-03-13 20:47:32
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')
print(train_data.info())

#**********************************************************************
age = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
age_notnull = age.loc[(train_data.Age.notnull())]
age_isnull = age.loc[(train_data.Age.isnull())]
X = age_notnull.values[:,1:]
Y = age_notnull.values[:,0]
rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
predictAges = rfr.predict(age_isnull.values[:,1:])
train_data.loc[(train_data.Age.isnull()),'Age'] = predictAges

train_data.loc[train_data['Sex']=='male','Sex'] = 0
train_data.loc[train_data['Sex']=='female','Sex'] = 1

train_data['Embarked'] = train_data['Embarked'].fillna('S')

train_data.drop(['Cabin'],axis=1,inplace=True)
# deceased就是1-survived
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)

train_data.info()
#***************************************************************************
#先只选取6个标签
dataset_X = train_data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
#用于在训练的时候，根据存活与每个维度里面存/亡概率大小的关系，优化网络
dataset_Y = train_data[['Deceased','Survived']]

#训练集中划分一部分作为x与对应y的验证集
X_train,X_val,Y_train,Y_val = train_test_split(dataset_X.values,dataset_Y.values,
                                                test_size = 0.2,random_state = 42)

x = tf.placeholder(tf.float32,shape = [None,6],name = 'input')
y = tf.placeholder(tf.float32,shape = [None,2],name = 'label')

#获取乘客个数，用于添加bias
passenger_Num = x.shape.as_list()
#每个标签对应survived/deceased的权重
weights = tf.Variable(tf.random_normal([6,2]),name = 'weights')
#bias = tf.Variable(tf.zeros([2000]),name = 'bias')
#根据用户的各个标签值，算出对应的survived和deceased参数，可不可以添加一个bias怎么设置？
z = tf.matmul(x,weights)
#将两个参数值转化为0~1的概率
y_pred = tf.nn.softmax(z)
# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z))
#对比概率更大的索引值(二维数组，每个维度只有0或1)，如果概率大的缩引相同，则判断正确
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
#统计准确率
acc_op = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
# 步长
train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

#下面开始训练，epoch为30次
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(30):
        total_loss = 0.
        for i in range(len(X_train)):
            feed_dict = {x: [X_train[i]],y:[Y_train[i]]}
            _,loss = sess.run([train_step,cost],feed_dict=feed_dict)
            total_loss +=loss
        print('Epoch: %4d, total loss = %.12f' % (epoch,total_loss))
        if (epoch+1) % 10 == 0:
            accuracy = sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
            print("Accuracy on validation set: %.9f" % accuracy)
    print('training complete!')

    accuracy = sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
    print("Accuracy on validation set: %.9f" % accuracy)
    pred = sess.run(y_pred,feed_dict={x:X_val})
    correct = (np.equal(np.argmax(pred,1),np.argmax(Y_val,1))).astype(np.int32)
    numpy_accuracy = np.mean(correct.astype(np.float32))#这里correct的平均值就是正确率
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)

    # 读测试数据, 测试数据的清洗和训练数据一样，两者可以共同完成
    test_data = pd.read_csv('test.csv')  

    #数据清洗, 数据预处理  
    test_data.loc[test_data['Sex']=='male','Sex'] = 0
    test_data.loc[test_data['Sex']=='female','Sex'] = 1 

    age = test_data[['Age','Sex','Parch','SibSp','Pclass']]
    age_notnull = age.loc[(test_data.Age.notnull())]
    age_isnull = age.loc[(test_data.Age.isnull())]
    X = age_notnull.values[:,1:]
    Y = age_notnull.values[:,0]
    rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(age_isnull.values[:,1:])
    test_data.loc[(test_data.Age.isnull()),'Age'] = predictAges

    test_data['Embarked'] = test_data['Embarked'].fillna('S')

    test_data.drop(['Cabin'],axis=1,inplace=True)

    #特征选择  
    X_test = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]  
    #预测  根据构建好的网络参数，算出预测值
    predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1) #取索引，对应为存亡

    #保存结果  
    submission = pd.DataFrame({"PassengerId": test_data["PassengerId"],  "Survived": predictions })  
    submission.to_csv("titanic-submission.csv", index=False)
