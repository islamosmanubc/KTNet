
""" Trainer for pre-train phase. """
import os
import numpy as np
import tensorflow as tf

from tqdm import trange
from data_generator.pre_data_generator import PreDataGenerator
from models.pre_KTNet import MakePreModel
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class preKTNetTrainer:
    """The class that contains the code for the pre-train phase"""
    def __init__(self):
        
        # This class defines the pre-train phase trainer
        print('Generating pre-training classes')
        
        # Generate Pre-train Data Tensors
        pre_train_data_generator = PreDataGenerator()
        pretrain_input, pretrain_label = pre_train_data_generator.make_data_tensor()
        pre_train_input_tensors = {'pretrain_input': pretrain_input, 'pretrain_label': pretrain_label}

        # Build Pre-train Model
        self.model = MakePreModel()
        self.model.construct_pretrain_model(input_tensors=pre_train_input_tensors)
        self.model.pretrain_summ_op = tf.summary.merge_all()

        # Start the TensorFlow Session
        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()

            gpu_config.gpu_options.allow_growth = True

            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

        # Initialize and Start the Pre-train Phase
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()
        self.pre_train()

    def pre_train(self):
        # Load Parameters from FLAGS
        pretrain_iterations = FLAGS.pretrain_iterations
        weights_save_dir_base = FLAGS.pretrain_dir
        pre_save_str = FLAGS.pre_string

        # Build Pre-train Log Folder
        weights_save_dir = os.path.join(weights_save_dir_base, pre_save_str)
        if not os.path.exists(weights_save_dir):
            os.mkdir(weights_save_dir)
        pretrain_writer = tf.summary.FileWriter(weights_save_dir, self.sess.graph)
        pre_lr = FLAGS.pre_lr

        print('Start pre-train phase')
        print('Pre-train Hyper parameters: ' + pre_save_str)


        e = 1
        weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(e)), \
                        allow_pickle=True, encoding="latin1").tolist()
        for key in weights.keys():
            if key == 'b5':
                w = weights[key][:64]
            elif key == 'w5':
                
                w = weights[key][:,:64]
            else:
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
        print('Pretrain weights loaded, saving init weights')

        # Start the iterations
        for itr in range(20000):
            # Generate the Feed Dict and Run the Optimizer
            feed_dict = {self.model.pretrain_lr: pre_lr}
            input_tensors = [self.model.pretrain_op, self.model.pretrain_summ_op]
            input_tensors.extend([self.model.pretrain_task_loss, self.model.pretrain_task_accuracy])
            result = self.sess.run(input_tensors, feed_dict)

            # Print Results during Training
            if (itr!=0) and itr % FLAGS.pre_print_step == 0:
                print_str = '[*] Pre Loss: ' + str(result[-2]) + ', Pre Acc: ' + str(result[-1])
                print(print_str)

            # Write the TensorFlow Summery
            if itr % FLAGS.pre_sum_step == 0:
                pretrain_writer.add_summary(result[1], itr+e)

            # Decrease the Learning Rate after Some Iterations
            if (itr!=0) and itr % FLAGS.pre_lr_dropstep == 0:
                pre_lr = pre_lr * 0.5
                if FLAGS.pre_lr_stop and pre_lr < FLAGS.min_pre_lr:
                    pre_lr = FLAGS.min_pre_lr

            # Save Pre-train Model
            if (itr!=0) and itr % FLAGS.pre_save_step == 0:
                print('Saving pretrain weights to npy')
                weights = self.sess.run(self.model.weights)
                np.save(os.path.join(weights_save_dir, "weights_{}.npy".format(itr+e+1)), weights)
