#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline models
"""

from tensorflow.keras.layers import Flatten,Dense,\
    Input,Dropout, Conv1D, AvgPool1D
from tensorflow.keras.models import Model  
from tensorflow.keras import optimizers,layers,losses


#------------------------------------------------------------------------------
# LSTM
#------------------------------------------------------------------------------

def model_lstm(lr=1e-4,N=64,inshape=40,nclass=12):
    in1=Input(shape=(inshape,1))
    x=layers.LSTM(N, activation='tanh')(in1) 
    x=Dropout(0.1)(x)
    x=Dense(128, activation='relu')(x) 
    out1=Dense(nclass, activation='softmax')(x)
    model=Model(inputs=in1,outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
              loss=losses.categorical_crossentropy,
              metrics=['acc'])
    return model

#------------------------------------------------------------------------------
#CONV1D
#------------------------------------------------------------------------------

def model_conv1D(lr=1e-4,N=64,inshape=40,nclass=12):
    in1=Input(shape=(inshape,1))
    x=Conv1D(N, 3,padding='same',activation='relu')(in1)  
    x=Conv1D(N, 3,padding='same',activation='relu')(x)  
      
    x=layers.Flatten()(x)
    x=Dropout(0.1)(x)
    x=Dense(128, activation='relu')(x)    
    out1=Dense(nclass, activation='softmax')(x)
    model=Model(inputs=in1,outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
              loss=losses.categorical_crossentropy,
              metrics=['acc'])
    return model


#------------------------------------------------------------------------------
# DENSE
#------------------------------------------------------------------------------

def model_dense(lr=1e-4,N=64,inshape=40,nclass=12):
    in1=Input(shape=(inshape,1))
    x=Dense(N,activation='relu')(in1)  
    x=Dense(N,activation='relu')(x)  
    x=layers.Flatten()(x)
    x=Dropout(0.1)(x)
    x=Dense(128, activation='relu')(x)    
    out1=Dense(nclass, activation='softmax')(x)
    model=Model(inputs=in1,outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
              loss=losses.categorical_crossentropy,
              metrics=['acc'])
    return model

#------------------------------------------------------------------------------
# BASE LINE models
#------------------------------------------------------------------------------
def model_conv1D_large(nfeat=40,lr=1e-2,nclass=12):     
    in1=Input(shape=(nfeat,1))
    x=Conv1D(64, 3,padding='same' ,activation='relu')(in1) 
    x=Conv1D(64, 3,padding='same' ,activation='relu')(x)   
    x=AvgPool1D()(x)
    x=Conv1D(128,5,padding='same' ,activation='relu')(x)
    x=AvgPool1D()(x)
    x=Conv1D(256,7,padding='same'  ,activation='relu')(x)
    x=AvgPool1D()(x)
    x=Conv1D(512,9,padding='same' ,activation='relu')(x)
    x=Flatten()(x)   
    x=Dropout(0.4)(x)
    x=Dense(512, activation='relu')(x)
    output=Dense(nclass, activation='softmax')(x)    
    model = Model(inputs=in1, outputs=output)
    
    #opt=optimizers.SGD(lr)
    opt=optimizers.Adam(lr)
    # opt=optimizers.RMSprop(lr)
    # opt=optimizers.Adam(lr)
    #opt=optimizers.Nadam(2e-2)
    model.compile(optimizer=opt,#Adam(lr=1e-2),
              loss=losses.categorical_crossentropy,
              metrics=['acc'])    
    return model

def model_conv1D_binary(nfeat=32,lr=1e-2,nclass=12):     
    in1=Input(shape=(nfeat,1))
    x=Conv1D(64, 3,padding='same' ,activation='relu')(in1) 
    x=Conv1D(64, 3,padding='same' ,activation='relu')(x)   
    x=AvgPool1D()(x)
    x=Conv1D(128,5,padding='same' ,activation='relu')(x)
    x=AvgPool1D()(x)
    x=Conv1D(256,7,padding='same'  ,activation='relu')(x)
    x=AvgPool1D()(x)
    x=Conv1D(512,9,padding='same' ,activation='relu')(x)
    x=Flatten()(x)   
    x=Dropout(0.2)(x)
    x=Dense(512, activation='relu')(x)
    output=Dense(nclass, activation='sigmoid')(x)    
    model = Model(inputs=in1, outputs=output)
    
    #opt=optimizers.SGD(lr)
    # opt=optimizers.Adam(lr)
    opt=optimizers.RMSprop(lr)
    # opt=optimizers.Adam(lr)
    #opt=optimizers.Nadam(2e-2)
    model.compile(optimizer=opt,#Adam(lr=1e-2),
              loss=losses.binary_crossentropy,
              metrics=['acc'])    
    return model
