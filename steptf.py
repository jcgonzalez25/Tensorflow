
import tensorflow as tf
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import sys
import os

def read_data(fname):

   it = pd.read_csv(fname,header=None)
   x = []
   y = []
   for team in it.values:
      x.append(float(team[0]))
      y.append(float(team[1]))
   return x,y

if __name__ == "__main__":

   best = 1e9

   xdata,ydata = read_data('data.csv')

   xmin = min(xdata)   # for plotting
   xmax = max(xdata)

   print(xdata)
   print(ydata)
   N = len(xdata)

   x = tf.placeholder(tf.float32)
   y = tf.placeholder(tf.float32)
   w = tf.Variable(tf.random_normal(()))
   b = tf.Variable(25.0 + tf.random_normal(()))

   rate = 0.001
   pred = w*x+b                     # model that predicts y from x
   loss = tf.square(y - pred)
   opti = tf.train.GradientDescentOptimizer(rate).minimize(loss)
   init = tf.global_variables_initializer()

   sess = tf.Session()

   sess.run(init)

   plt.scatter(xdata,ydata)

   for i in range(1000):
      k = rnd.randrange(N)
      sess.run(opti,feed_dict={x:xdata[k],y:ydata[k]})
      wval, bval = sess.run([w,b])
      print("w = {0:12.6f}, b = {1:12.6f}".format(wval,bval))
      plt.clf()
      plt.scatter(xdata,ydata)
      plt.plot([xmin,xmax],[wval*xmin+bval,wval*xmax+bval])
      plt.pause(0.10)

   plt.show()
