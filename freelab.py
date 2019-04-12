#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: ZhaoSong
@LastEditors: ZhaoSong
@Date: 2019-04-09 11:31:32
@LastEditTime: 2019-04-12 11:20:05
'''
import tensorflow as tf
import numpy as np
# test1
class FizzBuzz():
    def __init__(self, length=30):    # FizzBuzz函数初始化，定义变量
        self.length = length  # 程序需要执行的序列长度
        self.array = tf.Variable([str(i) for i in range(1, length+1)], dtype=tf.string, trainable=False)  # 最后程序返回的结果
        self.graph = tf.while_loop(self.cond, self.body, [1, self.array],)   # 对每一个值进行循环判断

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)    

    def cond(self, i, _):
        return (tf.less(i, self.length+1)) # 判断是否是最后一个值

    def body(self, i, _):   
        flow = tf.cond(    # tf.cond相当于一个if操作，为真则执行前者，false则执行后者  每次都进行一个flow
            tf.equal(tf.mod(i, 15), 0),  # 如果值能被 15 整除，那么就把该位置赋值为 FizzBuzz
                lambda: tf.assign(self.array[i - 1], 'FizzBuzz'),

                lambda: tf.cond(tf.equal(tf.mod(i, 3), 0), # 如果值能被 3 整除，那么就把该位置赋值为 Fizz
                        lambda: tf.assign(self.array[i - 1], 'Fizz'),
                        lambda: tf.cond(tf.equal(tf.mod(i, 5), 0),   # 如果值能被 5 整除，那么就把该位置赋值为 Buzz
                                lambda: tf.assign(self.array[i - 1], 'Buzz'),
                                lambda: self.array  # 最后返回的结果
                )
            )
        )
        return (tf.add(i, 1), flow)
# test2
class LinearSearch():
    def __init__(self, array, x):
        self.x = tf.constant(x)
        self.array = tf.constant(array)
        self.length = len(array)
        self.graph = tf.while_loop(self.cond, self.body, [0, self.x, False])

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, _, is_found):
        return tf.logical_and(tf.less(i, self.length), tf.logical_not(is_found))

    def body(self, i, _, is_found):
        return tf.cond(tf.equal(self.array[i], self.x),
                    lambda: (i, self.array[i], True),
                    lambda: (tf.add(i, 1), -1, False))
# test3
class BubbleSort():
    def __init__(self, array):
        self.i = tf.constant(0)
        self.j = tf.constant(len(array)-1)
        self.array = tf.Variable(array, trainable=False)
        self.length = len(array)

        cond = lambda i, j, _: tf.less(i-1, self.length-1)
        self.graph = tf.while_loop(cond, self.outer_loop, loop_vars=[self.i, self.j, self.array])

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def outer_loop(self, i, j, _):
        cond = lambda i, j, _: tf.greater(j, i)
        loop = tf.while_loop(cond, self.inner_loop, loop_vars=[i, self.length-1, self.array])
        return tf.add(i, 1), loop[1], loop[2]

    def inner_loop(self, i, j, _):
        body = tf.cond(tf.greater(self.array[j-1], self.array[j]),
                    lambda: tf.scatter_nd_update(self.array, [[j-1],[j]], [self.array[j],self.array[j-1]]),
                    lambda: self.array)
        return i, tf.subtract(j, 1), body       

if __name__ == '__main__':
   e= np.arange(10)
   print(e)
   e= e.reshape(1,2,5)
   print(e)
   e=np.squeeze(e)
   print(e)