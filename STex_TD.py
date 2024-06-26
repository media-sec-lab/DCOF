#This code is to achieve TD-PHCM to STEX Data Set Classification

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
from scipy import ndimage
from pandas import DataFrame
from skimage import io
from skimage import feature


#Parameter definition
#Definition of parameters, including Batch Size, Learning Rate, etc. Batch size and STEX class have equal
BATCH_SIZE = 476
NUM_LABELS = 476
IMAGE_SIZE=128
NUM_ITER =800000
NUM_SHOWTRAIN = 476*4 #show result eveary epoch 
NUM_SHOWTEST = 100
BN_DECAY = 0.95
UPDATE_OPS_COLLECTION = 'Discriminative_update_ops'
NUM_CHANNEL = 3
LEARNING_RATE =0.01
LEARNING_RATE_DECAY = 0.6
MOMENTUM = 0.9
decay_step = 80
is_train = True
path = 'data/STex'


#Gauss_num is the number of Gaussian kernel in TD-PHCMs,dim represents the dimension from which the co-occurrence matrix is to be extracted
Bin=3
Gauss_num=Bin*Bin
in_chancel=16
dim=2

#Read data in STex dataset randomly
fileList = []
for (dirpath,dirnames,filenames) in os.walk(path+'/0'):  
    fileList = filenames
  
random.seed(1234)
random.shuffle(fileList)
list_sta=np.zeros([BATCH_SIZE,8],dtype=np.int16)
for k in range(8):
    list_sta[:,k]=k

#The initial TD-PHCM
Mean_Value_Min=0
Mean_Value_Max=1
Standard_Deviation=0.6
u=-np.linspace(Mean_Value_Min, Mean_Value_Max, num=Bin,dtype=np.float32)
w1_init_L=np.zeros([dim-1,1,in_chancel,Gauss_num],dtype=np.float32)
w2_init_L=np.zeros([dim,1,in_chancel,Gauss_num],dtype=np.float32)

w1_init_R=np.zeros([1,dim-1,in_chancel,Gauss_num],dtype=np.float32)
w2_init_R=np.zeros([1,dim,in_chancel,Gauss_num],dtype=np.float32)




w1_init_L[0,0,:,:]=1/Standard_Deviation
w2_init_L[0,0,:,:]=1/Standard_Deviation


w1_init_R[0,0,:,:]=1/Standard_Deviation
w2_init_R[0,0,:,:]=1/Standard_Deviation
bias1_init=np.zeros([in_chancel*Gauss_num],np.float32)
bias2_init=np.zeros([in_chancel*Gauss_num],np.float32)
j=0
for i1 in range(Bin):
    for i2 in range(Bin):
        bias1_init[j]=u[i1]/(Standard_Deviation*Standard_Deviation)
        bias2_init[j]=u[i2]/(Standard_Deviation*Standard_Deviation)
        j=j+1

              



x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32,shape=[BATCH_SIZE,NUM_LABELS])
is_train = tf.placeholder(tf.bool,name='is_train')



with tf.variable_scope("Group1") as scope:
    
    #Linear convolution layer, used to generate feature graphs
    kernel0 = tf.Variable( tf.random_normal( [3,3,3,in_chancel],mean=0.0,stddev=0.01 ),name="kernel0" )  

    conv0 = tf.nn.conv2d(x/255, kernel0, [1,2,2,1], padding='SAME',name="conv0"  )
    
    
   
   
    

with tf.variable_scope("Group2") as scope:
    #Extract the co-occurrence matrix in the vertical direction, and extract Gauss_num co-occurrence elements from each feature map
   
    kernel1=tf.Variable( w1_init_L,name="kernel1" )  
    conv1=tf.nn.depthwise_conv2d(conv0[:,0:64-1,:,:], kernel1, [1,1,1,1], padding='VALID',name="conv3_1"  )
    bias1=tf.Variable(bias1_init,dtype=tf.float32,name="bias1")
    Y1= tf.nn.bias_add(conv1,bias1)
    
    kernel2=tf.Variable( w2_init_L,name="kernel1" )  
    conv2=tf.nn.depthwise_conv2d(conv0, kernel2, [1,1,1,1], padding='VALID',name="conv3_1"  )
    bias2=tf.Variable(bias1_init,dtype=tf.float32,name="bias1")
    Y2= tf.nn.bias_add(conv2,bias2)
    

    
    
  
    
  
    
    conv_Y1=tf.square(Y1)
    conv_Y2=tf.square(Y2)

   
    exp_input=conv_Y1+conv_Y2
    
    exp_input=tf.multiply(-0.5,exp_input)
    exp_result=tf.exp(exp_input) 
    exp_result=exp_result
    Group4_output1_a=tf.reduce_sum(exp_result,[1,2])
    
    
with tf.variable_scope("Group3") as scope:
   #Extract the horizontal co-occurrence matrix, and extract Gauss_num co-occurrence elements from each feature map
    kernel1=tf.Variable( w1_init_R,name="kernel1" )  
    conv1=tf.nn.depthwise_conv2d(conv0[:,:,0:64-1,:], kernel1, [1,1,1,1], padding='SAME',name="conv3_1"  )
    bias1=tf.Variable(bias1_init,dtype=tf.float32,name="bias1")
    Y1= tf.nn.bias_add(conv1,bias1)
    
    kernel2=tf.Variable( w2_init_R,name="kernel1" )  
    conv2=tf.nn.depthwise_conv2d(conv0, kernel2, [1,1,1,1], padding='VALID',name="conv3_1"  )
    bias2=tf.Variable(bias1_init,dtype=tf.float32,name="bias1")
    Y2= tf.nn.bias_add(conv2,bias2)
    

    
    
  
    
  
    
    conv_Y1=tf.square(Y1)
    conv_Y2=tf.square(Y2)

   
    exp_input=conv_Y1+conv_Y2
    
    exp_input=tf.multiply(-0.5,exp_input)
    exp_result=tf.exp(exp_input) 
    exp_result=exp_result
    Group4_output1_b=tf.reduce_sum(exp_result,[1,2])
with tf.variable_scope('Group4') as scope:
    #The extracted co-occurrence elements are concat and linear calssifer is input after entering BN layer
    Group4_output1=tf.concat([Group4_output1_a,Group4_output1_b],axis=-1)
    Group4_output= slim.layers.batch_norm(Group4_output1,is_training=is_train,updates_collections=None,decay=0.05) 
    weights1 = tf.Variable(tf.random_normal( [2*Gauss_num*in_chancel,476],mean=0.0,stddev=0.01 ),name="weights1" )
    bias1 = tf.Variable( tf.random_normal([476],mean=0.0,stddev=0.01),name="bias1" )
    y_ = tf.matmul(Group4_output, weights1) + bias1      
        
        

vars = tf.trainable_variables()
params = [v for v in vars if (v.name.startswith('Group1/') or  v.name.startswith('Group2/') or v.name.startswith('Group3/') or v.name.startswith('Group4/')) ]

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc',accuracy)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y,logits=y_))
tf.summary.scalar('loss',loss)
global_step = tf.Variable(0,trainable = False)
decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step, decay_step, LEARNING_RATE_DECAY, staircase=True)
opt = tf.train.AdamOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,var_list=params)

variables_averages = tf.train.ExponentialMovingAverage(0.95)
variables_averages_op = variables_averages.apply(tf.trainable_variables())

data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE,NUM_LABELS])

for i in range(BATCH_SIZE):
    data_y[i,i]=1

saver = tf.train.Saver()
merged = tf.summary.merge_all()
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
Max_test=np.zeros([1])
with tf.Session(config = config) as sess:
     writer = tf.summary.FileWriter("logs/logs_aug_1/train/shi_aug_s0.4_0731",sess.graph)
     tf.global_variables_initializer().run()
     #saver.restore(sess,"/home/wwh/program/python/tensorflow/shicnn/2017.6/shi_saver/suniward40/shi_june_1_200000.ckpt")
     summary = tf.Summary()	
     count = 0

     
     
         
                

     for i in range(1,NUM_ITER+1):

         
         for j in range(476):
             if count%8==0:
                count=count%8
                for k in range(476):
                     random.seed(i+k)
                     random.shuffle(list_sta[k,:])        
             img=io.imread(path+'/'+str(j)+'/'+fileList[list_sta[j,count]])
             des=np.random.rand(1)

             data_x[j,:,:,:]= img.astype(np.float32)
         count=count+1
         _,temp,l = sess.run([opt,accuracy,loss],feed_dict={x:data_x,y:data_y,is_train:True})
         
         if i%100==0:  
             summary.ParseFromString(sess.run(merged,feed_dict={x:data_x,y:data_y,is_train:True}))
             writer.add_summary(summary, i)
         if i%100==0:  
             print ('shi_aug_s0.4: batch result')
             print ('epoch:', i)
             print ('loss:', l)
             print ('accuracy:', temp)
             print (' ')
#         if i%(10*1280)==0:
#            saver.save(sess,'UIUC_H'+str(i)+'.ckpt')        
            
         if i%NUM_SHOWTEST==0:
            result1 = np.array([]) #accuracy for training set
            num = i/NUM_SHOWTEST - 1
            train_count = 0
            while train_count<8:
                for j in range(476):
                    image_sta=io.imread(path+'/'+str(j)+'/'+fileList[train_count])                      
                    data_x[j,:,:,:] = image_sta.astype(np.float32)
                train_count = train_count+1
                c1,temp1 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                result1 = np.insert(result1,0,temp1)
                summary.value.add(tag='val_acc', simple_value=np.mean(result1))
                writer.add_summary(summary, i)
            print ('train accuracy:', np.mean(result1))
            result2 = np.array([]) #accuracy for testing set
            test_count = 8
            while test_count<16:
                for j in range(476):
                    image_sta=io.imread(path+'/'+str(j)+'/'+fileList[test_count])                      
                    data_x[j,:,:,:] = image_sta.astype(np.float32)
                       
                test_count = test_count+1
                c2,temp2 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                result2= np.insert(result2,0,temp2)
                summary.value.add(tag='test_acc', simple_value=np.mean(result2))
                writer.add_summary(summary, i)
            print ('Testing :', np.mean(result2))
            if Max_test<np.mean(result2):
                Max_test=np.mean(result2)
            print ('MAX Testing :',Max_test)
            print (' ')
