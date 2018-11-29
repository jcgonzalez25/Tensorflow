import tensorflow as tf
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class FileReader:
  def __init__(self, csvFile):
    self.csvFile = csvFile
    self.y       = []
    self.x1      = []
    self.x2      = []
    self.rowSize = 0
    self.read()
  def read(self):
    it = pd.read_csv(self.csvFile,header=None,skiprows=1)
    y  = []
    x1 = []
    x2 = []
    self.rowSize = it.shape[0]
    for data in it.values:
      y.append(float(data[0]))
      x1.append(float(data[1]))
      x2.append(float(data[2]))
  def getRow(self,rowNum):
    return self.y[rowNum],self.x1,self.x2

class Plotter:
  def __init__(self,TestingData,TrainingData):
    self.TestingData  = TestingData
    self.TrainingData = TrainingData
      
class model:
  def __init__(self)
    self.x1    = tf.placeholder(tf.float32)
    self.x2    = tf.placeholder(tf.float32)
    self.y     = tf.placeholder(tf.float32)
    self.w1    = tf.Variable(tf.random_normal(()))
    self.w2    = tf.Variable(tf.random_normal(()))
    self.b1    = tf.Variable(tf.random_normal(()))
    self.b2    = tf.Variable(tf.random_normal(()))

    self.rate = 0.001
    self.pred = (self.w1 * self.x1 + self.b1)+(self.w2 * self.x2 + self.b2)
    self.loss = tf.square(self.y - self.pred)
    self.opti = tf.train.GradientDescentOptimizer(self.rate).minimize(self.loss)
    self.init = tf.global_variables_initializer()
  def Train(self,x1,x2):
    self.sess = tf.Session()
    sess.run(self.init)
    
    for i in range(1000)
      x1.getRow(i)

if __name__ == "__main__":
  TestingData  = FileReader("testing.csv")
  TrainingData = FileReader("training.csv")
#  Plotter      = Plotter(TestingData,TrainingData)
  
  
  
  
   
