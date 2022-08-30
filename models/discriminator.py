

""" Models for meta-learning. """
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block
from models.resnet48 import Models

FLAGS = flags.FLAGS

def MakeMetaModel():
    """The function to make meta model.
    Arg:
      Meta-train model class.
    """
   
    class MetaModel(Models):
        """The class for the meta models. This class is inheritance from Models, so some variables are in the Models class."""
        def construct_model(self):
            """The function to construct meta-train model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32) # episode train images
            self.inputb = tf.placeholder(tf.float32) # episode test images
            self.labeldisc = tf.placeholder(tf.float32) # episode test labels

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()

                _,emb_outputa = self.forward_resnet_disc(self.inputa, weights, ss_weights,dweights, reuse=tf.AUTO_REUSE) # Embed episode train
                _,emb_outputb = self.forward_resnet_disc(self.inputb, weights, ss_weights,dweights, reuse=True) # Embed episode test



                margin = 0.5
                self.distance = tf.sqrt(tf.reduce_sum(tf.pow(emb_outputa - emb_outputb, 2), 1, keepdims=True))
                similarity = self.labeldisc * tf.square(self.distance)
                dissimilarity = (1 - self.labeldisc) * tf.square(tf.maximum((margin - self.distance), 0))
                self.lossdisc = tf.reduce_mean(dissimilarity + similarity) / 2
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.metatrain_op = optimizer.minimize(self.lossdisc, var_list=list(dweights.values()))

                   

        def construct_test_model(self):
            """The function to construct meta-test model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
                self.weights = weights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()

                # Load test base epoch number from FLAGS
                num_updates = FLAGS.test_base_epoch_num

                def task_metalearn(inp, reuse=True):
                    """The function to process one episode in a meta-batch.
                    Args:
                      inp: the input episode.
                      reuse: whether reuse the variables for the normalization.
                    Returns:
                      A serious outputs like losses and accuracies.
                    """
                    # Seperate inp to different variables
                    inputa, inputb, labela, labelb = inp
                    # Generate empty list to record accuracies
                    accb_list = []

                    # Embed the input images to embeddings with ss weights
                    emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                    emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                    # This part is similar to the meta-train function, you may refer to the comments above
                    outputa = self.forward_fc(emb_outputa, fc_weights)
                    lossa = self.loss_func(outputa, labela)
                    grads = tf.gradients(lossa, list(fc_weights.values()))
                    gradients = dict(zip(fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                        self.update_lr*gradients[key] for key in fc_weights.keys()]))
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)
                    
                    for j in range(num_updates - 1):
                        lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                        grads = tf.gradients(lossa, list(fast_fc_weights.values()))
                        gradients = dict(zip(fast_fc_weights.keys(), grads))
                        fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))
                        outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                        accb_list.append(accb)

                    lossb = self.loss_func(outputb, labelb)

                    task_output = [lossb, accb, accb_list]

                    return task_output

                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossesb, accsb, accsb_list = result

            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]

    return MetaModel()
