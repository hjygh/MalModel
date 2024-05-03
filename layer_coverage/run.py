import argparse

from datetime import datetime
from keras.models import model_from_json, load_model, save_model
import os
import numpy as np

from utils import load_MNIST, load_CIFAR, load_IMAGENET
from utils import filter_val_set, get_trainable_layers
from neuron_cov import NeuronCoverage
from model import Model

# example: python run.py -M model/classifier_graph -DS traffic -A nc -L -2 -s 224 -ba 50 -th 0.5 > traffic-0.5.txt

def parse_arguments():
    # define the program description
    text = 'Coverage Analyzer for DNNs'
    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.")
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        or cifar10 or imagenet).", choices=["mnist","cifar10","imagenet","diyface","diytraffic"])
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to measure coverage", choices=['nc'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)
    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)
    parser.add_argument("-b", "--begin", help="input img begin", type=int)
    parser.add_argument("-e", "--end_num", help="input img num", type=int)
    parser.add_argument("-s", "--img_size", help="img size", type=int)
    parser.add_argument("-ba", "--batch_size", help="batch size", type=int)
    parser.add_argument("-th", "--threshold", help="threshold", type=float)

    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    print("begin")
    args = parse_arguments()
    model_path     = args['model'] if args['model'] else 'neural_networks/LeNet5'
    dataset        = args['dataset'] if args['dataset'] else 'mnist'
    approach       = args['approach'] if args['approach'] else 'nc'
    selected_class = args['class'] if not args['class']==None else -1 #ALL CLASSES


    ####################
    # 0) Load data
    isimgnet = 0
    if dataset == 'mnist':
        X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
        img_rows, img_cols = 28, 28
    elif dataset == 'imagenet':
        isimgnet = 1
        labelpath = 'img\caffe_ilsvrc12\\val.txt'
        imgpath = 'img\ILSVRC2012_img_val\\'
        beg = args['begin'] if args['begin'] else 0
        end = args['end_num'] if args['end_num'] else 100
        imgs, labels = load_IMAGENET(imgpath,labelpath,beg,end)
        X_train = imgs
        X_test = imgs
        print('load imagenet successfully.')
        print(X_train.shape)
    elif dataset == 'face':
        print("face dataset")
        imgpath = 'data/face/'
        imgs = []
        imgslist = os.listdir(imgpath)
        for f in imgslist:
            imgs.append(imgpath+f)
        imgs = np.array(imgs)
        X_train = imgs
        X_test = imgs
        print('load face dataset successfully.')
        print(X_train.shape)
    elif dataset == 'traffic':
        print("traffic dataset")
        imgpath = 'data/traffic/'
        imgs = []
        imgslist = os.listdir(imgpath)
        for f in imgslist:
            imgs.append(imgpath+f)
        imgs = np.array(imgs)
        X_train = imgs
        X_test = imgs
        print('load traffic dataset successfully.')
        print(X_train.shape)
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()
        img_rows, img_cols = 32, 32


    if not selected_class == -1:
        X_train, Y_train = filter_val_set(selected_class, X_train, Y_train) #Get training input for selected_class
        X_test, Y_test = filter_val_set(selected_class, X_test, Y_test) #Get testing input for selected_class



    ####################
    # 1) Setup the model
    model_name = model_path.split('/')[-1]
    flag = 0

    try:
        json_file = open(model_path + '.json', 'r') #Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')

        # Compile the model before using
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        try:
            model = load_model(model_path + '.h5')
        except:
            flag = 1
            model = Model(model_path+'.pb','tensorflow')

    # 2) Load necessary information
    skip_layers = [0] 
    if flag==0:
        trainable_layers = get_trainable_layers(model)
        non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
        print(len(model.layers))
        print(model.layers[12])
        print(model.summary())
        print('Trainable layers: ' + str(trainable_layers))
        print('Non trainable layers: ' + str(non_trainable_layers))

        experiment_folder = 'experiments'

        #Investigate the penultimate layer
        subject_layer = args['layer'] if not args['layer'] == None else -1
        print('1: ',subject_layer)
        subject_layer = trainable_layers[subject_layer]
        print('2: ',subject_layer)
        
        for idx, lyr in enumerate(model.layers):
            if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

        print("Skipping layers:", skip_layers)

    ####################
    # 3) Analyze Coverages
    if approach == 'nc':
        batch = args['batch_size'] if args['batch_size'] else 25
        size = args['img_size'] if args['img_size'] else 224
        th = args['threshold'] if args['threshold'] else 0.75
        nc = NeuronCoverage(isimgnet, model, threshold=th, skip_layers = skip_layers, batch = batch, size = size) 
        coverage, _, _, _, _ = nc.test(X_test)
        print("Your test set's coverage is: ", coverage)

        nc.set_measure_state(nc.get_measure_state())
    else :
        print("other methods, like idc, kmnc and so on.")

