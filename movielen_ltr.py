#!/usr/bin/env python
# coding=UTF-8
'''
@Description: tensorflow test
@Author: ZhaoSong
@LastEditors: ZhaoSong
@Date: 2019-03-05 13:19:08
@LastEditTime: 2019-04-15 20:33:48
'''
import pandas as pd 
import numpy as np
import tensorflow as tf
from collections import defaultdict
import random

def data_processor():
    """
    read ratings and movies dataset"""
    ratings_df = pd.read_csv('E:ml/ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('E:ml/ml-latest-small/movies.csv')
    # add column movierow as movies index
    movies_df['MovieRow'] = movies_df.index
    # processing feature extraction
    movies_df = movies_df[['MovieRow','movieId','title']]
    movies_df.to_csv('E:ml/ml-latest-small/moviesProcessed.csv',index=False, header = True, encoding = 'utf-8')

    ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    return ratings_df

#***************************init*************************************#
ratings_df = data_processor()
userNo = ratings_df['userId'].max()+1
movieNo = ratings_df['MovieRow'].max()+1
rating = np.zeros((movieNo,userNo))

batch_size = 50
label_size = movieNo

flag = 0
#get columns_num of the merged table
ratings_df_length = np.shape(ratings_df)[0]
#遍历矩阵，将电影的评分填入表中
for index,row in ratings_df.iterrows():
    rating[int(row['MovieRow']), int(row['userId'])] = row['rating']
    flag += 1
    if flag%1000 ==0:
        print('processed %d, %d left' %(flag,ratings_df_length-flag))
#record：indicater whether the film is rated 
record = rating > 0
record = np.array(record, dtype = int)
#**********************************************************************#

def p_top1(mat):
    """
    The top one probability indicates the probability of 
    an item being ranked in the top position for a given ranking list. 
    """
    return (tf.exp(mat)/tf.reduce_sum(tf.exp(mat)))

def d_sigmoid(x):
    return (tf.matmul(tf.sigmoid(x),tf.constant(1)-tf.sigmoid(x)))

'''A test for batch'''
def get_batch(train_data, batch_size, label_size):
    """ 
    to devide your to be trained data into batches.
    Args:
    train_data: the dataset you want to split
    batch_size: the length of batches
    label_size: the width of your label"""
    data_length = train_data.shape[0]
    iterations = (data_length - 1 )/ (batch_size*label_size)
    #round_data_len = iterations * batch_size * label_size
    
    for i in range(iterations):
        batch = train_data[:,i*label_size:(i+1)*label_size]
        yield batch

# List-wise Learning to Rank with Matrix Factorization 
def ListRank_MF(batch_size, item_count, hidden_dim):
    """
    batch_size:每个batch用户数
    item_count:电影总数
    hidden_dim:矩阵分解的特征维度"""
    # 传入的batch
    u = tf.placeholder(tf.int32, shape = [batch_size,item_count], name = input)
    # 分解为用户U和物品V
    user_u = tf.Variable(tf.random_normal([batch_size,hidden_dim],stddev = 0.35), name = 'user_u')
    item_v = tf.Variable(tf.random_normal([item_count,hidden_dim],stddev = 0.35), name = 'item_v') 
    pred_mat = tf.matmul(user_u,item_v,transpose_b=True)
    # 获取真实和预测值的top1概率
    ratings_top1 = p_top1(u)
    pred_top1 = p_top1(tf.sigmoid(pred_mat))
    # 损失函数和对应梯度
    cost = tf.reduce_sum(-tf.reduce_sum(tf.multiply(ratings_top1,tf.log(pred_top1))*record)) + 1/2 * (user_u** 2 + item_v ** 2)
    grad_u = tf.reduce_sum(tf.matmul((tf.log(pred_top1)-ratings_top1),tf.matmul(d_sigmoid(pred_mat),item_v))* record) + user_u
    grad_v = tf.reduce_sum(tf.matmul((tf.log(pred_top1)-ratings_top1),tf.matmul(d_sigmoid(pred_mat),user_u))* record) + item_v
    # 梯度下降
    learning_rate = 0.001
    new_U = user_u.assign(user_u - learning_rate * grad_u)
    new_V = item_v.assign(item_v - learning_rate * grad_v)

    return pred_mat, cost, new_U, new_V

with tf.Session() as sess:
    batch = get_batch(rating,batch_size,label_size)
    total_batch = int(userNo/batch_size)

    pred_mat, cost, new_U, new_V = ListRank_MF(userNo,movieNo,20)

    tf.global_variables_initializer().run()
    for Epoch in range(0, 4):
        total_loss = 0
        for k in range(total_batch):
            feed_dict = {batch[k]}
            loss, _, _ = sess.run([cost, new_U, new_V], feed_dict=feed_dict)
            total_loss += loss
            if (k+1)*10 % total_batch ==0:
                print('Epoch: %d, total loss = %d' % (Epoch+1,total_loss))

    pred_ratings = sess.run(pred_mat)
    print(pred_ratings)
    print("为用户推荐：")
    p = np.squeeze(pred_ratings)
    p[np.argsort(p)[:-5]] = 0
    for index in range(len(p)):
        if p[index] != 0:
            print (index, p[index])

