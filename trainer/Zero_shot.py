""" Trainer for meta-learning. """
import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import time

from tqdm import trange
from data_generator.meta_data_generator import MetaDataGenerator
from models.meta_model import MakeMetaModel
from tensorflow.python.platform import flags
from sklearn.metrics.pairwise import cosine_similarity
from utils.misc import process_batch
from random import randint
from sklearn.cluster import KMeans
from sklearn import cluster, mixture

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

class ZeroShot:
    """The class that contains the code for the meta-train and meta-test."""
    def __init__(self):
        
        # Remove the saved datalist for a new experiment
        os.system('rm -r ./logs/processed_data/*')
        data_generator = MetaDataGenerator()
        
        if FLAGS.load_saved_weights:
            random.seed(7)
        data_generator.generate_data(data_type='test')

        if FLAGS.metatrain:
            # Build model for meta-train phase
            print('Building meta-train model')
            self.model = MakeMetaModel()
            self.model.construct_model_disc()
            print('Meta-train model is built')
            # Start tensorflow session          
            self.start_session()
            # Generate data for meta-train phase
            if FLAGS.load_saved_weights:
                random.seed(5) 
            data_generator.generate_data(data_type='train')
            if FLAGS.load_saved_weights:
                random.seed(7)
            data_generator.generate_data(data_type='test')
            if FLAGS.load_saved_weights:
                random.seed(9)
            data_generator.generate_data(data_type='val')
            
        else:
            # Build model for meta-test phase
            print('Building meta-test mdoel')
            self.model = MakeMetaModel()
            
            self.model.construct_test_zero()


            self.model.summ_op = tf.summary.merge_all()
            print('Meta-test model is built')
            # Start tensorflow session
            self.start_session()
            # Generate data for meta-test phase
            if FLAGS.load_saved_weights:
                random.seed(7)
            data_generator.generate_data(data_type='test')
        # Load the experiment setting string from FLAGS
        exp_string = FLAGS.exp_string
        exp_string = 'metatrain logs'
        # Global initialization and starting queue
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        if FLAGS.metatrain:
            # Process initialization weights for meta-train
            init_dir = FLAGS.logdir_base + 'init_weights/'
            if not os.path.exists(init_dir):
                os.mkdir(init_dir)
            pre_save_str = FLAGS.pre_string
            pre_save_str = 'logs'
            this_init_dir = init_dir + pre_save_str + '.pre_iter(' + str(FLAGS.pretrain_iterations) + ')/'
            if not os.path.exists(this_init_dir):
                # If there is no saved initialization weights for meta-train, load pre-train model and save initialization weights
                os.mkdir(this_init_dir)
                if FLAGS.load_saved_weights:
                    print('Loading downloaded pretrain weights')
                    weights = np.load('logs/download_weights/weights.npy', allow_pickle=True, encoding="latin1").tolist()
                else:
                    print('Loading pretrain weights')
                    weights_save_dir_base = FLAGS.pretrain_dir
                    weights_save_dir = os.path.join(weights_save_dir_base, pre_save_str)
                    weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(FLAGS.pretrain_iterations)), \
                        allow_pickle=True, encoding="latin1").tolist()
                bais_list = [bais_item for bais_item in weights.keys() if '_bias' in bais_item]
                # Assign the bias weights to ss model in order to train them during meta-train
                for bais_key in bais_list:
                    self.sess.run(tf.assign(self.model.ss_weights[bais_key], weights[bais_key]))
                # Assign pretrained weights to tensorflow variables
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                print('Pretrain weights loaded, saving init weights')
                # Load and save init weights for the model
                new_weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights = self.sess.run(self.model.fc_weights)
                np.save(this_init_dir + 'weights_init.npy', new_weights)
                np.save(this_init_dir + 'ss_weights_init.npy', ss_weights)
                np.save(this_init_dir + 'fc_weights_init.npy', fc_weights)
            else:
                # If the initialization weights are already generated, load the previous saved ones
                # This process is deactivate in the default settings, you may activate this for ablative study
                print('Loading previous saved init weights')
                weights = np.load(this_init_dir + 'weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                dweights = np.load(this_init_dir + 'dweights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                ss_weights = np.load(this_init_dir + 'ss_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                dss_weights = np.load(this_init_dir + 'dss_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load(this_init_dir + 'fc_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in dweights.keys():
                    self.sess.run(tf.assign(self.model.dweights[key], dweights[key]))
                for key in ss_weights.keys():
                    self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
                for key in dss_weights.keys():
                    self.sess.run(tf.assign(self.model.dss_weights[key], dss_weights[key]))
                for key in fc_weights.keys():
                    if key == 'b5':
                        w = fc_weights[key][:5]
                        self.sess.run(tf.assign(self.model.fc_weights[key], w))
                    elif key == 'w5':
                 
                        w = fc_weights[key][:,:5]
                        self.sess.run(tf.assign(self.model.fc_weights[key], w))
                print('Init weights loaded')
        else:
            # Load the saved meta model for meta-test phase
            if FLAGS.load_saved_weights:
                # Load the downloaded weights
                weights = np.load('./logs/download_weights/weights.npy', allow_pickle=True, encoding="latin1").tolist()
                ss_weights = np.load('./logs/download_weights/ss_weights.npy', allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load('./logs/download_weights/fc_weights.npy', allow_pickle=True, encoding="latin1").tolist()
            else:
                # Load the saved weights of meta-train
                exp_string = 'metalearninglogs'
                    
                    
                self.weights = np.load(FLAGS.logdir + '/' + exp_string +  '/weights_mini.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                self.dweights = np.load(FLAGS.logdir + '/' + exp_string +  '/dweights_mini.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                self.ss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/ss_weights_mini.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                self.fc_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/fc_weights_mini.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                
            # Assign the weights to the tensorflow variables
            for key in self.weights.keys():
                self.sess.run(tf.assign(self.model.weights[key],self.weights[key]))
            for key in self.dweights.keys():
                self.sess.run(tf.assign(self.model.dweights[key], self.dweights[key]))
            for key in self.ss_weights.keys():
                self.sess.run(tf.assign(self.model.ss_weights[key], self.ss_weights[key]))
            print('Weights loaded')
            if FLAGS.load_saved_weights:
                print('Meta test using downloaded model')
            else:
                print('Test iter: ' + str(FLAGS.test_iter))

        if FLAGS.metatrain:
            self.train_disc(data_generator)
        else:
            #self.test(data_generator)
            self.testKTNET(data_generator)

    def start_session(self):
        """The function to start tensorflow session."""
        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

    def copyweights(self):
        tf.global_variables_initializer().run()
        for key in self.weights.keys():
            self.sess.run(tf.assign(self.model.weights[key], self.weights[key]))
        for key in self.dweights.keys():
            self.sess.run(tf.assign(self.model.dweights[key], self.dweights[key]))
        for key in self.ss_weights.keys():
            self.sess.run(tf.assign(self.model.ss_weights[key], self.ss_weights[key]))
    def re_initialize(self,classes,p):
        self.model.construct_test_model(classes,p+1)
        self.copyweights()

    def test(self, data_generator):
        """The function for the meta-test phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Set meta-test episode number
        NUM_TEST_POINTS = 10
        # Load the experiment setting string from FLAGS
        exp_string = FLAGS.exp_string
        exp_string = 'testresults'
        print('Start meta-test phase')
        np.random.seed(1)
        # Generate empty list to record accuracies
        metaval_accuracies = []
        disc_accuracies = []
        discwithout_accuracies = []
        metawithout_accuracies = []
        itr = FLAGS.test_base_epoch_num-1
        testsize = FLAGS.way_num*FLAGS.metatest_epite_sample_num
        # Load data for meta-test
        data_generator.load_data(data_type='test')

        detected = 0
        tdetected = 0
        fdetected = 0

        for test_idx in trange(NUM_TEST_POINTS):

            indices=[]
            visited = np.zeros(testsize)
            currentclasses=2
            labels = {}
            # Load one episode for meta-test
            this_episode = data_generator.load_episode(index=test_idx, data_type='test')
            inputa = this_episode[0][np.newaxis, :]
            labela = this_episode[1][np.newaxis, :]
            inputb = this_episode[2][np.newaxis, :]
            labelb = this_episode[3][np.newaxis, :]

            inputa = []
            labela = []
            detected = detected +1
            tdetected = tdetected + 1
            value = randint(0, testsize-1)
            indices.append(value)
            inputa = [inputb[0][value]]
            inputa = np.array(inputa)
            h, w = inputa.shape
            inputa = np.reshape(inputa,[1,h,w])
            labels[0] = labelb[0][value]
            labela = [[1]]
            labela = np.array(labela)
            h, w = labela.shape
            labela = np.reshape(labela,[1,h,w])

            while(True):

                feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                        self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: 0.0}
                result = self.sess.run([self.model.d], feed_dict)
                
                distance = result[0]

                def PickUnknown():
                    least = 9999
                    idx = -1
                    for w in range(testsize):
                        maxd = np.max(distance[:,w])
                        if(maxd<0.3):
                            if(maxd < least):
                                least = maxd
                                idx=w
                    return idx
                
                xclass = -1 
                xclass = PickUnknown()
           
                if(xclass != -1):
                    indices.append(xclass)
                    currentclasses = len(indices)
                    b = inputb[0][xclass]
                    a = inputa[0]
                    a = np.vstack([a,b])
                    newlabel = np.zeros(currentclasses)
                    newlabel[-1] = 1
                    alreadyexist = False
                    for r in range(currentclasses-1):
                        if(np.argmax(labelb[0][xclass]) == np.argmax(labels[r])):
                            alreadyexist = True
                    if(alreadyexist==False):
                        tdetected = tdetected + 1
                    else:
                        fdetected = fdetected + 1

                    detected = detected + 1

                    labels[currentclasses-1] = labelb[0][xclass]

                    newlabels=[]
                    for i in range(currentclasses-1):
                        la = labela[0][i]
                        la = np.append(la,0)
                        newlabels.append(la)
                    
                    labela = np.array(newlabels)

                    la = labela
                    la = np.vstack([la,newlabel])
                    inputa= a

                    labela = la
                    h, w = labela.shape
                    labela = np.reshape(labela,[1,h,w])
                    h, w = inputa.shape
                    inputa = np.reshape(inputa,[1,h,w])
                else:
                    break
            nways = 0
            currentclasses = len(indices)
            if(currentclasses < FLAGS.way_num):
                nways = FLAGS.way_num
                diff = FLAGS.way_num-currentclasses
                for y in range(diff):
                    
                    newlabels=[]
                    for i in range(len(labela[0])):
                        la = labela[0][i]
                        la = np.append(la,0)
                        newlabels.append(la)
                    labela = np.array(newlabels)
                    h, w = labela.shape
                    labela = np.reshape(labela,[1,h,w])
            else:
                nways = currentclasses
                diff = currentclasses-FLAGS.way_num
                for y in range(diff):
                    newlabels=[]
                    for i in range(len(labelb[0])):
                        la = labelb[0][i]
                        la = np.append(la,0)
                        newlabels.append(la)
                    labelb = np.array(newlabels)
                    h, w = labelb.shape
                    labelb = np.reshape(labelb,[1,h,w])


            def updateLastLayer(currentclasses,p):
                self.re_initialize(currentclasses,p)
            
            updateLastLayer(nways,test_idx)

            for u in range(testsize):
            
            
                feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                    self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: 0.0}
                result = self.sess.run([self.model.metaval_total_accuracies,self.model.d,self.model.updatepredictions], feed_dict)

                

                distance = result[1]
                pred = result[2][itr]

                def ComputeAccBasedOnSim():
                    accc = np.zeros(testsize)
                    for w in range(testsize):
                        dist = distance[:,w]
                        n = np.argmax(dist)
                        simlabel = np.argmax(labela[0][n])
                        if(simlabel == np.argmax(labelb[0][w])):
                            accc[w]=1
                    return accc

                if(u==0):
                    metawithout_accuracies.append(result[0])
                    discwithout_accuracies.append(np.mean(ComputeAccBasedOnSim()))


                def PickSupport():
                    supports = []
                    for w in range(len(distance[0])):
                        dis = distance[:,w]
                        n = np.argmax(dis)
                        avg = np.mean(dis)
                        simlabel = np.argmax(labela[0][n])
                        alabel = labelb[0][w]
                        mtlabel = np.argmax(pred[0][w,:])
                        if(mtlabel==simlabel):
                            res = abs(pred[0][w,mtlabel]-np.mean(pred[0][w,:]))
                            if(res > 0.4):
                                if(np.max(dis) > 0.755):
                                    if(visited[w]==0):
                                        supports.append(w)
                                        visited[w] = 1
                    return supports        
                
                knownclasses = PickSupport()
                if(len(knownclasses)==0):
                    metaval_accuracies.append(result[0])
                    disc_accuracies.append(np.mean(ComputeAccBasedOnSim()))
                    break

                a = inputa[0]
                la = labela[0]
                        
                exist = False

                for y in range(len(knownclasses)):
                    b = inputb[0][knownclasses[y]]
                    
                    newlabel = np.zeros(nways)
                    ay = np.argmax(labelb[0][knownclasses[y]])
                    
                    newlabel[ay] = 1
                    la = np.vstack([la,newlabel])

                    a = np.vstack([a,b])
                

                inputa= a

                

                labela = la
                h, w = labela.shape
                labela = np.reshape(labela,[1,h,w])
                h, w = inputa.shape
                inputa = np.reshape(inputa,[1,h,w])

        metaval_accuracies = np.array(metaval_accuracies)
        metawithout_accuracies = np.array(metawithout_accuracies)
        disc_accuracies = np.array(disc_accuracies)
        discwithout_accuracies = np.array(discwithout_accuracies)

        means1 = np.mean(metaval_accuracies, 0)
        means2 = np.mean(metawithout_accuracies, 0)
        means3 = np.mean(disc_accuracies)
        means4 = np.mean(discwithout_accuracies)

        stds1 = np.std(metaval_accuracies, 0)
        stds2 = np.std(metawithout_accuracies, 0)
        stds3 = np.std(disc_accuracies)
        stds4 = np.std(discwithout_accuracies)

        ci951 = 1.96*stds1/np.sqrt(NUM_TEST_POINTS)
        ci952 = 1.96*stds2/np.sqrt(NUM_TEST_POINTS)
        ci953 = 1.96*stds3/np.sqrt(NUM_TEST_POINTS)
        ci954 = 1.96*stds4/np.sqrt(NUM_TEST_POINTS)

        # Print the meta-test results
        print('Test accuracies and confidence intervals')
        print((means1, ci951))

        # Save the meta-test results in the csv files
        if not FLAGS.load_saved_weights:
            out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_Algo1_' + str(FLAGS.test_iter) + '.csv'
            out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_Algo1_' + str(FLAGS.test_iter) + '.pkl'
            with open(out_pkl, 'wb') as f:
                pickle.dump({'mses': metaval_accuracies}, f)
            with open(out_filename, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['update'+str(i) for i in range(len(means1))])
                writer.writerow(means1)
                writer.writerow(means2)
                writer.writerow([str(means3) for i in range(len(means1))])
                writer.writerow([str(means4) for i in range(len(means1))])
                writer.writerow(stds1)
                writer.writerow(stds2)
                writer.writerow([str(stds3) for i in range(len(means1))])
                writer.writerow([str(stds4) for i in range(len(means1))])
                writer.writerow(ci951)
                writer.writerow(ci952)
                writer.writerow([str(ci953) for i in range(len(means1))])
                writer.writerow([str(ci954) for i in range(len(means1))])

                writer.writerow([str(detected) for i in range(len(means1))])
                writer.writerow([str(tdetected) for i in range(len(means1))])
                writer.writerow([str(fdetected) for i in range(len(means1))])


    def testKTNET(self, data_generator):
        """The function for the meta-test phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Set meta-test episode number
        NUM_TEST_POINTS = 1
        # Load the experiment setting string from FLAGS
        exp_string = FLAGS.exp_string
        exp_string = 'revisedresults'
        print('Start meta-test phase')
        np.random.seed(1)
        # Generate empty list to record accuracies
        metaval_accuracies = []
        disc_accuracies = []
        discwithout_accuracies = []
        metawithout_accuracies = []
        itr = FLAGS.test_base_epoch_num-1
        testsize = FLAGS.way_num*FLAGS.metatest_epite_sample_num
        # Load data for meta-test
        data_generator.load_data(data_type='test')

        detected = 0
        tdetected = 0
        fdetected = 0

        for test_idx in trange(NUM_TEST_POINTS):

            inputa = []
            labela = []
            inputb = []
            labelb = []
            inputadisc = []
            labeldisc = []
            inputa_ssl = []
            inputb_ssl = []
            label_ssl= []
            templabel=[]
            indices=[]
            visited = np.zeros(testsize)
            currentclasses=2
            labels = {}
            # Load one episode for meta-test
            this_episode = data_generator.load_episode(index=test_idx, data_type='test')
            inputa.append(this_episode[0])
            labela.append(this_episode[1])
            inputb.append(this_episode[2])
            labelb.append(this_episode[3])
            labeldisc.append(this_episode[4])
            inputb_ssl.append(this_episode[5])
            label_ssl.append(this_episode[6])
            templabel.append(this_episode[7])

            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)
            labeldisc = np.array(labeldisc)
            inputb_ssl = np.array(inputb_ssl)
            label_ssl = np.array(label_ssl)

            inputa = []
            labela = []
            detected = detected +1
            tdetected = tdetected + 1
            value = randint(0, testsize-1)
            indices.append(value)
            inputa = [inputb[0][value]]
            inputa = np.array(inputa)
            h, w = inputa.shape
            inputa = np.reshape(inputa,[1,h,w])
            labels[0] = labelb[0][value]
            labela = [[1]]
            labela = np.array(labela)
            h, w = labela.shape
            labela = np.reshape(labela,[1,h,w])
            
            start = time.time()

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                    self.model.labela: labela, self.model.labelb: labelb,self.model.labeldisc: labeldisc, self.model.meta_lr: 0.0, \
                    self.model.b_ssl:inputb_ssl,self.model.label_ssl:label_ssl}
            result = self.sess.run([self.model.d,self.model.updated_tr_weights], feed_dict)

            tr = result[1]
            tr['transformer'] = tr['transformer'].reshape([1,1,512,512])
            for key in tr.keys():
                self.sess.run(tf.assign(self.model.tr_weights[key], tr[key]))

            for qqq in range(FLAGS.way_num-1):

                feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                    self.model.labela: labela, self.model.labelb: labelb,self.model.labeldisc: labeldisc, self.model.meta_lr: 0.0, \
                    self.model.b_ssl:inputb_ssl,self.model.label_ssl:label_ssl}
                result = self.sess.run([self.model.sim], feed_dict)
                distance = result[0]

                def PickUnknown():
                    least = 99996
                    idx = -1
                    for w in range(testsize):
                        maxd = np.max(distance[:,w])
                        if(maxd<0.5):
                            if(maxd < least):
                                least = maxd
                                idx=w
                    return idx
                
                xclass = -1 
                xclass = PickUnknown()
           
                if(xclass != -1):
                    indices.append(xclass)
                    currentclasses = len(indices)
                    b = inputb[0][xclass]
                    a = inputa[0]
                    a = np.vstack([a,b])
                    newlabel = np.zeros(currentclasses)
                    newlabel[-1] = 1
                    alreadyexist = False
                    for r in range(currentclasses-1):
                        if(np.argmax(labelb[0][xclass]) == np.argmax(labels[r])):
                            alreadyexist = True
                    if(alreadyexist==False):
                        tdetected = tdetected + 1
                    else:
                        fdetected = fdetected + 1

                    detected = detected + 1

                    labels[currentclasses-1] = labelb[0][xclass]

                    newlabels=[]
                    for i in range(currentclasses-1):
                        la = labela[0][i]
                        la = np.append(la,0)
                        newlabels.append(la)
                    
                    labela = np.array(newlabels)

                    la = labela
                    la = np.vstack([la,newlabel])
                    inputa= a

                    labela = la
                    h, w = labela.shape
                    labela = np.reshape(labela,[1,h,w])
                    h, w = inputa.shape
                    inputa = np.reshape(inputa,[1,h,w])
                else:
                    break
                
            
            end1 = time.time()
            nways = 0
            currentclasses = len(indices)
            if(currentclasses < FLAGS.way_num):
                nways = FLAGS.way_num
                diff = FLAGS.way_num-currentclasses
                for y in range(diff):
                    
                    newlabels=[]
                    for i in range(len(labela[0])):
                        la = labela[0][i]
                        la = np.append(la,0)
                        newlabels.append(la)
                    labela = np.array(newlabels)
                    h, w = labela.shape
                    labela = np.reshape(labela,[1,h,w])
            else:
                nways = currentclasses
                diff = currentclasses-FLAGS.way_num
                for y in range(diff):
                    newlabels=[]
                    for i in range(len(labelb[0])):
                        la = labelb[0][i]
                        la = np.append(la,0)
                        newlabels.append(la)
                    labelb = np.array(newlabels)
                    h, w = labelb.shape
                    labelb = np.reshape(labelb,[1,h,w])

            
            
            pred_temp = []
            testsize = 1
            start1 = time.time()
            for u in range(testsize):
            
            
                feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                    self.model.labela: labela, self.model.labelb: labelb,self.model.labeldisc: labeldisc, self.model.meta_lr: 0.0, \
                    self.model.b_ssl:inputb_ssl,self.model.label_ssl:label_ssl}
                result = self.sess.run([self.model.metaval_total_accuracies,self.model.sim,self.model.updatepredictions], feed_dict)


                distance = result[1]
                pred = result[2][itr]

               

                def ComputeAccBasedOnSim():
                    accc = np.zeros(testsize)
                    for w in range(testsize):
                        dist = distance[:,w]
                        n = np.argmax(dist)
                        simlabel = np.argmax(labela[0][n])
                        if(simlabel == np.argmax(labelb[0][w])):
                            accc[w]=1
                    return accc

                if(u==0):
                    metawithout_accuracies.append(result[0])
                    discwithout_accuracies.append(np.mean(ComputeAccBasedOnSim()))


                def PickSupport():
                    supports = []
                    for w in range(len(distance[0])):
                        dis = distance[:,w]
                        n = np.argmax(dis)
                        avg = np.mean(dis)
                        simlabel = np.argmax(labela[0][n])
                        alabel = labelb[0][w]
                        mtlabel = np.argmax(pred[0][w,:])
                        if(mtlabel==simlabel):
                            res = abs(pred[0][w,mtlabel]-np.mean(pred[0][w,:]))
                            if(res > 0.4):
                                if(np.max(dis) > 0.65):
                                    if(visited[w]==0):
                                        supports.append(w)
                                        visited[w] = 1
                    return supports        
                
                knownclasses = PickSupport()
                if(len(knownclasses)==0):
                    metaval_accuracies.append(result[0])
                    disc_accuracies.append(np.mean(ComputeAccBasedOnSim()))
                    break

                a = inputa[0]
                la = labela[0]
                        
                exist = False

                for y in range(len(knownclasses)):
                    b = inputb[0][knownclasses[y]]
                    
                    newlabel = np.zeros(nways)
                    ay = np.argmax(labelb[0][knownclasses[y]])
                    
                    newlabel[ay] = 1
                    la = np.vstack([la,newlabel])

                    a = np.vstack([a,b])
                

                inputa= a

                

                labela = la
                h, w = labela.shape
                labela = np.reshape(labela,[1,h,w])
                h, w = inputa.shape
                inputa = np.reshape(inputa,[1,h,w])
                pred_temp = pred
            
            end = time.time()
        # Calculate the mean accuarcies and the confidence intervals
        c1 = end-start1
        c2 = end1-start
        print(f"Runtime of the program is {end - start1}")
        
        print(f"Runtime of the program is {end1 - start}")
        
        print(f"Runtime of the program is {c1 + c2}")
        metaval_accuracies = np.array(metaval_accuracies)
        metawithout_accuracies = np.array(metawithout_accuracies)
        disc_accuracies = np.array(disc_accuracies)
        discwithout_accuracies = np.array(discwithout_accuracies)

        means1 = np.mean(metaval_accuracies, 0)
        means2 = np.mean(metawithout_accuracies, 0)
        means3 = np.mean(disc_accuracies)
        means4 = np.mean(discwithout_accuracies)

        stds1 = np.std(metaval_accuracies, 0)
        stds2 = np.std(metawithout_accuracies, 0)
        stds3 = np.std(disc_accuracies)
        stds4 = np.std(discwithout_accuracies)

        ci951 = 1.96*stds1/np.sqrt(NUM_TEST_POINTS)
        ci952 = 1.96*stds2/np.sqrt(NUM_TEST_POINTS)
        ci953 = 1.96*stds3/np.sqrt(NUM_TEST_POINTS)
        ci954 = 1.96*stds4/np.sqrt(NUM_TEST_POINTS)

        # Print the meta-test results
        print('Test accuracies and confidence intervals')
        print((means1, ci951))

        # Save the meta-test results in the csv files
        if not FLAGS.load_saved_weights:
            out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'MINI_0shottest' + str(FLAGS.test_iter) + '.csv'
            out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_Algo1_' + str(FLAGS.test_iter) + '.pkl'
            with open(out_pkl, 'wb') as f:
                pickle.dump({'mses': metaval_accuracies}, f)
            with open(out_filename, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['update'+str(i) for i in range(len(means1))])
                writer.writerow(means1)
                writer.writerow(means2)
                writer.writerow([str(means3) for i in range(len(means1))])
                writer.writerow([str(means4) for i in range(len(means1))])
                writer.writerow(stds1)
                writer.writerow(stds2)
                writer.writerow([str(stds3) for i in range(len(means1))])
                writer.writerow([str(stds4) for i in range(len(means1))])
                writer.writerow(ci951)
                writer.writerow(ci952)
                writer.writerow([str(ci953) for i in range(len(means1))])
                writer.writerow([str(ci954) for i in range(len(means1))])

                writer.writerow([str(detected) for i in range(len(means1))])
                writer.writerow([str(tdetected) for i in range(len(means1))])
                writer.writerow([str(fdetected) for i in range(len(means1))])
