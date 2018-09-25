import matplotlib
matplotlib.use('Agg')
import os

import keras
import tensorflow as tf
from keras import backend as k
import os.path
import numpy as np
from os import mkdir,environ
import json
from keras.models import Sequential, load_model
from keras.layers import Bidirectional,LSTM, Dense, Activation, TimeDistributed, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import math
from tf_utils import normalize
from sklearn.utils import shuffle

framework = "keras"
draw_graph = 1
run_on_cpu = True

def customLoss(yTrue,yPred):
    return K.mean(K.abs(yTrue - yPred))

def loadConfig(configPath=None):
    if not configPath:
        cfg = {
            "lstm_size" : 4,
            "n_inputs" : 1,
            "learning_rate" : 0.001,
            "num_features" : 11,
            "drop_rate" : 0.5
        }
        print ("loaded default config")

    else:
        with open(configPath, "r") as configFile:
            json_str = configFile.read()
            cfg = json.loads(json_str)
            print (cfg)
    global LSTM_SIZE
    global N_INPUTS
    global INPUT_PATH
    global LEARNING_RATE
    global NUM_FEATURES
    global DROP_RATE

    LSTM_SIZE = cfg["lstm_size"]
    N_INPUTS = cfg["n_inputs"]
    LEARNING_RATE = cfg["learning_rate"]
    NUM_FEATURES = cfg["num_features"]
    DROP_RATE = cfg["drop_rate"]
    return cfg
def getNextRunDir(prefix):
    script_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(script_path, "lstm_runs",prefix)

    #Find the directory name in the series which is not used yet
    for num_run in range(0,500000):
        if not os.path.isdir(output_path+'_{}'.format(num_run)):
            mkdir(output_path+'_{}'.format(num_run))
            output_path = output_path+'_{}'.format(num_run)
            break
    return output_path

def printConfig(dir,cfg):
    str = json.dumps(cfg)
    with open(os.path.join(dir,"netConfig.json"),"wb") as f:
        f.write(str)


parser = argparse.ArgumentParser()
parser.add_argument("--infer",help="Input the model path")
args = parser.parse_args()

doLoadConfig = 0 # set this to "1" when doing hyperparameter search

if not args.infer:
    configPath = None
    if doLoadConfig == 1:
        configPath = "curRunConfig_lstm.json"
    if run_on_cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cfg = loadConfig(configPath)
    
    # Read the train and validation dataset
    train_path = 'RDF_40_normed_time.csv'
    train_df = pd.read_csv(train_path,delimiter=',',header=None)
    train_df = shuffle(train_df)
    print (train_df.iloc[0:5,:].values)
    Y_train = train_df.iloc[:,11].values
    X_train = train_df.iloc[:,0:11].values
    X_train = normalize(X_train)

    X_valid = X_train[0:345,:]
    X_train = X_train[345:,:]
    Y_valid = Y_train[0:345]
    Y_train = Y_train[345:]

    Y_train = np.reshape(Y_train, (Y_train.shape[0],1))
    Y_valid = np.reshape(Y_valid, (Y_valid.shape[0],1))
    X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
    X_valid = X_valid.reshape((X_valid.shape[0],1,X_valid.shape[1]))
    print ('Training data shape')
    print (X_train.shape)
    print ('Validation data shape')
    print (X_valid.shape)
    print (X_valid[0,:,:])
    
    # Construct the model
    
    model = Sequential()
    # Input Layer
    model.add(LSTM(NUM_FEATURES,return_sequences=False,input_shape=(1,NUM_FEATURES)))
    model.add(Activation('relu'))
    #model.add(Dropout(DROP_RATE))
    # Middle Layer
    #model.add(Bidirectional(LSTM(2,return_sequences=True)))
    #model.add(Activation('relu'))
    #model.add(Dropout(DROP_RATE))
    # Output Layer
    #model.add(LSTM(1))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(1))
    
    # If you want to load a good model for continuing training, use this command
    #model = load_model('lstm_runs/lstm_search_127/last.h5')
    
    # Compile the model
    model.compile(loss='mean_squared_error',
                    optimizer=keras.optimizers.Adam(lr=LEARNING_RATE))
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    output_dir = getNextRunDir('lstm_search')
    tbCallBack = keras.callbacks.TensorBoard(log_dir=output_dir, histogram_freq=1, write_grads=True, write_graph= False, write_images=True)
    checkpoint = keras.callbacks.ModelCheckpoint(output_dir+'/best.h5', monitor='val_loss',save_best_only=True)
    
    
    # Fit the model
    history = model.fit(X_train, Y_train, epochs=500, batch_size=8, validation_data =(X_valid, Y_valid), verbose=1,callbacks = [checkpoint])
    model.save(output_dir+'/last.h5')
    print('Results stored in {}'.format(output_dir))
    
    
    # Plot the validation loss
    if draw_graph:
        valid_predict = np.asarray(model.predict(X_valid))
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Hydraulic gradient_measured', fontsize = 15)
        ax.set_ylabel('Hydraulic gradient_predicted', fontsize = 15)
        ax.set_title('LSTM_Multiscale_Modeling', fontsize = 20)
        ax.scatter(Y_valid, valid_predict,c = 'b',s = 50)
        ax.grid()
        plt.savefig('LSTM_Multiscale_Modeling_Prediction')

        
# Testing part
if args.infer:
    start_time = time.time()
    model = load_model(args.infer) #custom_objects={'customLoss': customLoss}
    model.summary()
    test_path = 'RDF_40_test_normed.csv'
    test_df = pd.read_csv(test_path,delimiter=',', header=None)

    Y_test = test_df.iloc[:,11].values
    X_test = test_df.iloc[:,0:11].values
    X_test = normalize(X_test)
    
    Y_test = np.reshape(Y_test, (Y_test.shape[0],1))
    X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))
    
    print ("test dataset shape")
    print (X_test.shape)
    print (Y_test.shape)
    
    test_predict = np.asarray(model.predict(X_test))
    np.savetxt('lstm_test_predict.csv', test_predict, delimiter=',',header='Predicted')
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Y_test', fontsize = 15)
    ax.set_ylabel('Test_predict', fontsize = 15)
    ax.set_title('LSTM_Multiscale_Modeling', fontsize = 20)
    ax.scatter(Y_test, test_predict,c = 'b',s = 50)
    #plt.xlim([0.9,2.5])
    #plt.ylim([0.9,2.5])
    ax.grid()
    plt.savefig('LSTM_Multiscale_Modeling_Test_Prediction')


