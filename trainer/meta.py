""" Trainer for meta-learning. """
import time
import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf

from tqdm import trange
from data_generator.meta_data_generator import MetaDataGenerator
from models.meta_model import MakeMetaModel
from tensorflow.python.platform import flags
from sklearn.metrics.pairwise import cosine_similarity
from utils.misc import process_batch

FLAGS = flags.FLAGS

class MetaTrainer:
    """The class that contains the code for the meta-train and meta-test."""
    def __init__(self):
        
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Remove the saved datalist for a new experiment
        os.system('rm -r ./logs/processed_data/*')
        data_generator = MetaDataGenerator()

        if FLAGS.load_saved_weights:
                random.seed(5) 
        data_generator.generate_data(data_type='train')
        if FLAGS.load_saved_weights:
            random.seed(7)
        data_generator.generate_data(data_type='test')
        if FLAGS.load_saved_weights:
            random.seed(9)
        #data_generator.generate_data(data_type='val')

        if FLAGS.metatrain:
            # Build model for meta-train phase
            print('Building meta-train model')
            self.model = MakeMetaModel()
            self.model.construct_model_disc_KTNET()
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
            self.model.construct_test_disc_ktnet()
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
                ##dss_weights = np.load(this_init_dir + 'dss_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load(this_init_dir + 'fc_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in dweights.keys():
                    self.sess.run(tf.assign(self.model.dweights[key], dweights[key]))
                for key in ss_weights.keys():
                    self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
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
                weights = np.load(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(1) + '.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                dweights = np.load(FLAGS.logdir + '/' + exp_string +  '/dweights_' + str(1) + '.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                ss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(1) + '.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(1) + '.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                tr_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/tr_weights_' + str(1) + '.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
                dss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/dss_weights_' + str(1) + '.npy', \
                    allow_pickle=True, encoding="latin1").tolist()
            # Assign the weights to the tensorflow variables
            for key in weights.keys():
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
            for key in dweights.keys():
                self.sess.run(tf.assign(self.model.dweights[key], dweights[key]))
            for key in ss_weights.keys():
                self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
            for key in dss_weights.keys():
                self.sess.run(tf.assign(self.model.dss_weights[key], dss_weights[key]))
            for key in tr_weights.keys():
                self.sess.run(tf.assign(self.model.tr_weights[key], tr_weights[key]))
            for key in fc_weights.keys():
                if key == 'b5':
                    w = fc_weights[key][:5]
                    self.sess.run(tf.assign(self.model.fc_weights[key][:5], w))
                elif key == 'w5':
                    w = fc_weights[key][:,:5]
                    self.sess.run(tf.assign(self.model.fc_weights[key][:,:5], w))
            print('Weights loaded')
            if FLAGS.load_saved_weights:
                print('Meta test using downloaded model')
            else:
                print('Test iter: ' + str(FLAGS.test_iter))

        if FLAGS.metatrain:
            self.train_disc_ktnet(data_generator)
        else:
            self.test_disc_ktnet(data_generator)

    def start_session(self):
        """The function to start tensorflow session."""
        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

    def train(self, data_generator):
        """The function for the meta-train phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Load the experiment setting string from FLAGS 
        exp_string = FLAGS.exp_string
        exp_string = 'metalearninglogs'
        # Generate tensorboard file writer
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Start meta-train phase')
        # Generate empty list to record losses and accuracies
        loss_list, acc_list = [], []
        # Load the meta learning rate from FLAGS
        train_lr = FLAGS.meta_lr
        # Load data for meta-train and meta validation
        data_generator.load_data(data_type='train')
        data_generator.load_data(data_type='val')

        for train_idx in trange(FLAGS.metatrain_iterations):
            # Load the episodes for this meta batch
            inputa = []
            labela = []
            inputb = []
            labelb = []
            inputadisc = []
            labeldisc = []
            for meta_batch_idx in range(FLAGS.meta_batch_size):
                this_episode = data_generator.load_episode(index=train_idx*FLAGS.meta_batch_size+meta_batch_idx, data_type='train')
                inputa.append(this_episode[0])
                labela.append(this_episode[1])
                inputb.append(this_episode[2])
                labelb.append(this_episode[3])
                
            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)

            # Generate feed dict for the tensorflow graph
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: train_lr}

            # Set the variables to load from the tensorflow graph
            input_tensors = [self.model.metatrain_op] # The meta train optimizer
            input_tensors.extend([self.model.total_loss]) # The meta train loss
            input_tensors.extend([self.model.total_accuracy]) # The meta train accuracy

            # run this meta-train iteration
            result = self.sess.run(input_tensors, feed_dict)

            # record losses, accuracies and tensorboard
            
            #loss_list.append(result[1])
            loss_list.append(result[1])
            acc_list.append(result[2])
            #train_writer.add_summary(result[3], train_idx)

            # print meta-train information on the screen after several iterations
            if (train_idx!=0) and train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Acc:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list = [], []

            # Save the model during meta-teain
            if train_idx % FLAGS.meta_save_step == 0:
                #weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights = self.sess.run(self.model.fc_weights)
                #np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx) + '.npy', weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx) + '.npy', ss_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx) + '.npy', fc_weights)

            # Run the meta-validation during meta-train
            if (train_idx+1) % FLAGS.meta_val_print_step == 0:
                test_loss = []
                test_accs = []
                for test_itr in range(FLAGS.meta_intrain_val_sample):
                    this_episode = data_generator.load_episode(index=test_itr, data_type='val')
                    test_inputa = this_episode[0][np.newaxis, :]
                    test_labela = this_episode[1][np.newaxis, :]
                    test_inputb = this_episode[2][np.newaxis, :]
                    test_labelb = this_episode[3][np.newaxis, :]

                    test_feed_dict = {self.model.inputa: test_inputa, self.model.inputb: test_inputb, \
                        self.model.labela: test_labela, self.model.labelb: test_labelb, \
                        self.model.meta_lr: 0.0}
                    test_input_tensors = [self.model.total_loss, self.model.total_accuracy]
                    test_result = self.sess.run(test_input_tensors, test_feed_dict)
                    test_loss.append(test_result[0])
                    test_accs.append(test_result[1])
                    
                print_str = '[***] Val Loss:' + str(np.mean(test_loss)*FLAGS.meta_batch_size) + \
                    ' Val Acc:' + str(np.mean(test_accs)*FLAGS.meta_batch_size)
                print(print_str)

            # Reduce the meta learning rate to half after several iterations
            if (train_idx!=0) and train_idx % FLAGS.lr_drop_step == 0:
                train_lr = train_lr * FLAGS.lr_drop_rate
                if train_lr < 0.1 * FLAGS.meta_lr:
                    train_lr = 0.1 * FLAGS.meta_lr
                print('Train LR: {}'.format(train_lr))

        # Save the final model
        #weights = self.sess.run(self.model.weights)
        ss_weights = self.sess.run(self.model.ss_weights)
        fc_weights = self.sess.run(self.model.fc_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx+1) + '.npy', ss_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx+1) + '.npy', fc_weights)



    def train_disc(self, data_generator):
        """The function for the meta-train phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Load the experiment setting string from FLAGS 
        exp_string = FLAGS.exp_string
        exp_string = 'metalearninglogs'
        # Generate tensorboard file writer
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Start meta-train phase')
        # Generate empty list to record losses and accuracies
        loss_list, acc_list = [], []
        # Load the meta learning rate from FLAGS
        train_lr = FLAGS.meta_lr
        # Load data for meta-train and meta validation
        data_generator.load_data(data_type='train')
        data_generator.load_data(data_type='val')

        for train_idx in trange(FLAGS.metatrain_iterations):
            # Load the episodes for this meta batch
            inputa = []
            labela = []
            inputb = []
            labelb = []
            inputadisc = []
            labeldisc = []
            for meta_batch_idx in range(FLAGS.meta_batch_size):
                this_episode = data_generator.load_episode(index=train_idx*FLAGS.meta_batch_size+meta_batch_idx, data_type='train')
                inputa.append(this_episode[0])
                labela.append(this_episode[1])
                inputb.append(this_episode[2])
                labelb.append(this_episode[3])
                labeldisc.append(this_episode[4])

            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)
            labeldisc = np.array(labeldisc)

            # Generate feed dict for the tensorflow graph
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb,self.model.labeldisc:labeldisc, \
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: train_lr}

            # Set the variables to load from the tensorflow graph
            input_tensors = [self.model.emb_outputa] # The meta train optimizer 0
            input_tensors.extend([self.model.emb_outputb]) # The meta train loss 1
            input_tensors.extend([self.model.distance]) # The meta train loss 2
            input_tensors.extend([self.model.lossdisc]) # The meta train loss 3
            input_tensors.extend([self.model.metatrain_op]) # The meta train loss 4
            input_tensors.extend([self.model.d]) # The meta train loss 5

            # run this meta-train iteration
            result = self.sess.run(input_tensors, feed_dict)
            
            
            

            loss_list.append(result[3])
            
            d = result[2]
            acc=0.0
            for i in range(len(d)):
                if(d[i]>=0.5 and labeldisc[0][i]==0):
                    acc = acc + 1.0
                elif(d[i]<0.5 and labeldisc[0][i]==1):
                    acc = acc + 1.0
                    
            acc_list.append(acc/len(d))
            



            # print meta-train information on the screen after several iterations
            if (train_idx!=0) and train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Err:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list = [], []

            # Save the model during meta-teain
            if train_idx % FLAGS.meta_save_step == 0:
                ss_weights = self.sess.run(self.model.dweights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/dweights_' + str(train_idx) + '.npy', ss_weights)

            

            # Reduce the meta learning rate to half after several iterations
            if (train_idx!=0) and train_idx % FLAGS.lr_drop_step == 0:
                train_lr = train_lr * FLAGS.lr_drop_rate
                if train_lr < 0.1 * FLAGS.meta_lr:
                    train_lr = 0.1 * FLAGS.meta_lr
                print('Train LR: {}'.format(train_lr))

        # Save the final model
        ss_weights = self.sess.run(self.model.dweights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/dweights_' + str(train_idx+1) + '.npy', ss_weights)


    def train_disc_ktnet(self, data_generator):
        """The function for the meta-train phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Load the experiment setting string from FLAGS 
        exp_string = FLAGS.exp_string
        exp_string = 'metalearninglogs'
        # Generate tensorboard file writer
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Start meta-train phase')
        # Generate empty list to record losses and accuracies
        loss_list, acc_list = [], []
        # Load the meta learning rate from FLAGS
        train_lr = FLAGS.meta_lr
        # Load data for meta-train and meta validation
        data_generator.load_data(data_type='train')
        data_generator.load_data(data_type='val')

        for train_idx in trange(FLAGS.metatrain_iterations):
            # Load the episodes for this meta batch
            inputa = []
            labela = []
            inputb = []
            labelb = []
            inputadisc = []
            labeldisc = []
            inputa_ssl= []
            inputb_ssl = []
            label_ssl= []

            for meta_batch_idx in range(FLAGS.meta_batch_size):
                this_episode = data_generator.load_episode(index=train_idx*FLAGS.meta_batch_size+meta_batch_idx, data_type='train')
                inputa.append(this_episode[0])
                labela.append(this_episode[1])
                inputb.append(this_episode[2])
                labelb.append(this_episode[3])
                labeldisc.append(this_episode[4])
                inputb_ssl.append(this_episode[5])
                label_ssl.append(this_episode[6])

            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)
            labeldisc = np.array(labeldisc)
            inputb_ssl = np.array(inputb_ssl)
            label_ssl = np.array(label_ssl)

            # Generate feed dict for the tensorflow graph
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                self.model.labela: labela, self.model.labelb: labelb,self.model.labeldisc: labeldisc, self.model.meta_lr: train_lr, \
                self.model.b_ssl:inputb_ssl,self.model.label_ssl:label_ssl}

            # Set the variables to load from the tensorflow graph
            
            input_tensors = []
            input_tensors = [self.model.metatrain_op] # The meta train optimizer
            
            input_tensors.extend([self.model.total_loss]) # The meta train loss
            input_tensors.extend([self.model.similarity]) # The meta train accuracy
            

            # run this meta-train iteration
            result = self.sess.run(input_tensors, feed_dict)

            # record losses, accuracies and tensorboard
            
            loss_list.append(result[1])

            d = result[2][0]
            acc=0.0
            for i in range(len(d)):
                if(d[i]>=0.5 and labeldisc[0][i]==0):
                    acc = acc + 1.0
                elif(d[i]<0.5 and labeldisc[0][i]==1):
                    acc = acc + 1.0

            acc_list.append(1-(acc/len(d)))


            #train_writer.add_summary(result[3], train_idx)

            # print meta-train information on the screen after several iterations
            if (train_idx!=0) and train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Acc:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list = [], []

            # Save the model during meta-teain
            if train_idx % FLAGS.meta_save_step == 0:
                #weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.dss_weights)
                fc_weights = self.sess.run(self.model.tr_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/dss_weights_' + str(train_idx) + '.npy', ss_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/tr_weights_' + str(train_idx) + '.npy', fc_weights)

            
            # Reduce the meta learning rate to half after several iterations
            if (train_idx!=0) and train_idx % FLAGS.lr_drop_step == 0:
                train_lr = train_lr * FLAGS.lr_drop_rate
                if train_lr < 0.1 * FLAGS.meta_lr:
                    train_lr = 0.1 * FLAGS.meta_lr
                print('Train LR: {}'.format(train_lr))

        # Save the final model
        ss_weights = self.sess.run(self.model.dss_weights)
        fc_weights = self.sess.run(self.model.tr_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/dss_weights_' + str(train_idx+1) + '.npy', ss_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/tr_weights_' + str(train_idx+1) + '.npy', fc_weights)


    def test(self, data_generator):
        """The function for the meta-test phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Set meta-test episode number
        NUM_TEST_POINTS = 100
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
        for test_idx in trange(NUM_TEST_POINTS):

            visited = np.zeros(testsize)
            # Load one episode for meta-test
            this_episode = data_generator.load_episode(index=test_idx, data_type='test')
            inputa = this_episode[0][np.newaxis, :]
            labela = this_episode[1][np.newaxis, :]
            inputb = this_episode[2][np.newaxis, :]
            labelb = this_episode[3][np.newaxis, :]
           
           
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
                                if(np.max(dis) > 0.73):
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
                        
                for y in range(len(knownclasses)):
                    b = inputb[0][knownclasses[y]]
                    a = np.vstack([a,b])
                

                inputa= a

                
                la = labela[0]

                
                for y in range(len(knownclasses)):
                    newlabel = np.zeros(5)
                    ay = np.argmax(labelb[0][knownclasses[y]])
                    newlabel[ay] = 1
                    la = np.vstack([la,newlabel])

                labela = la
                h, w = labela.shape
                labela = np.reshape(labela,[1,h,w])
                h, w = inputa.shape
                inputa = np.reshape(inputa,[1,h,w])
                

        # Calculate the mean accuarcies and the confidence intervals
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


    


    def test_disc_ktnet(self, data_generator):
        """The function for the meta-test phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Set meta-test episode number
        NUM_TEST_POINTS = 1
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

            visited = np.zeros(testsize)
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
           
           
            start = time.time()
            for u in range(testsize):
                start1 = time.time()
            
            
                feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                    self.model.labela: labela, self.model.labelb: labelb,self.model.labeldisc: labeldisc, self.model.meta_lr: 0.0, \
                    self.model.b_ssl:inputb_ssl,self.model.label_ssl:label_ssl}


                result = self.sess.run([self.model.metaval_total_accuracies,self.model.d,self.model.updatepredictions, self.model.tr_weights, self.model.updated_tr_weights,\
                    ], feed_dict)

                

                distance = result[1][0]
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
                for y in range(len(knownclasses)):
                    b = inputb[0][knownclasses[y]]
                    a = np.vstack([a,b])

                inputa= a

                
                la = labela[0]

                
                for y in range(len(knownclasses)):
                    newlabel = np.zeros(5)
                    ay = np.argmax(labelb[0][knownclasses[y]])
                    newlabel[ay] = 1
                    la = np.vstack([la,newlabel])
                    
                labela = la
                h, w = labela.shape
                labela = np.reshape(labela,[1,h,w])
                h, w = inputa.shape
                inputa = np.reshape(inputa,[1,h,w])
                
        
        end = time.time()
        c1 = end-start1
        c2 = end-start
        print(f"Runtime of the program is {end - start1}")
        
        print(f"Runtime of the program is {end - start}")
        
        print(f"Runtime of the program is {c1 + c2}")
        # Calculate the mean accuarcies and the confidence intervals
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
            out_filename = FLAGS.logdir +'/'+ exp_string + '/' + '_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.csv'
            out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + '_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.pkl'
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


