

""" Models for meta-learning. """
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block
from models.resnet48 import Models
from sklearn.metrics.pairwise import cosine_similarity

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
            self.labela = tf.placeholder(tf.float32) # episode train labels
            self.labelb = tf.placeholder(tf.float32) # episode test labels

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()

                # Load base epoch number from FLAGS
                num_updates = FLAGS.train_base_epoch_num

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
                    # Generate empty list to record losses
                    lossa_list = [] # Base train loss list
                    lossb_list = [] # Base test loss list

                    # Embed the input images to embeddings with ss weights
                    emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse) # Embed episode train


                    emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True) # Embed episode test



                    # Run the first epoch of the base learning
                    # Forward fc layer for episode train
                    outputa = self.forward_fc(emb_outputa, fc_weights)
                    # Calculate base train loss
                    lossa = self.loss_func(outputa, labela)
                    # Record base train loss
                    lossa_list.append(lossa)
                    # Forward fc layer for episode test
                    outputb = self.forward_fc(emb_outputb, fc_weights)
                    # Calculate base test loss
                    lossb = self.loss_func(outputb, labelb)
                    # Record base test loss
                    lossb_list.append(lossb)
                    # Calculate the gradients for the fc layer
                    grads = tf.gradients(lossa, list(fc_weights.values()))
                    gradients = dict(zip(fc_weights.keys(), grads))
                    # Use graient descent to update the fc layer
                    fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                        self.update_lr*gradients[key] for key in fc_weights.keys()]))
              
                    for j in range(num_updates - 1):
                        # Run the following base epochs, these are similar to the first base epoch
                        lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                        lossa_list.append(lossa)
                        lossb = self.loss_func(self.forward_fc(emb_outputb, fast_fc_weights), labelb)
                        lossb_list.append(lossb)
                        grads = tf.gradients(lossa, list(fast_fc_weights.values()))
                        gradients = dict(zip(fast_fc_weights.keys(), grads))
                        fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))

                    # Calculate final episode test predictions
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    # Calculate the final episode test loss, it is the loss for the episode on meta-train 
                    final_lossb = self.loss_func(outputb, labelb)
                    # Calculate the final episode test accuarcy
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                    
                    #self.lossdisc = final_lossb
                    # Reorganize all the outputs to a list
                    task_output = [final_lossb, lossb_list, lossa_list, accb]

                    return task_output

                # Initial the batch normalization weights
                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                # Set the dtype of the outputs
                out_dtype = [tf.float32, [tf.float32]*num_updates, [tf.float32]*num_updates, tf.float32]

                # Run two episodes for a meta batch using parallel setting
                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                # Seperate the outputs to different variables
                lossb, lossesb, lossesa, accsb = result


            # Set the variables to output from the tensorflow graph
            self.total_loss = total_loss = tf.reduce_sum(lossb) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
            self.total_lossa = total_lossa = [tf.reduce_sum(lossesa[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.total_lossb = total_lossb = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]


            
            # Set the meta-train optimizer
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.metatrain_op = optimizer.minimize(self.total_loss, var_list=list(ss_weights.values()) + list(fc_weights.values()))

            # Set the tensorboard
            self.training_summaries = []
            self.training_summaries.append(tf.summary.scalar('Meta Train Loss', (total_loss / tf.to_float(FLAGS.metatrain_epite_sample_num))))
            self.training_summaries.append(tf.summary.scalar('Meta Train Accuracy', total_accuracy))
            for j in range(num_updates):
                self.training_summaries.append(tf.summary.scalar('Base Train Loss Step' + str(j+1), total_lossa[j]))
            for j in range(num_updates):
                self.training_summaries.append(tf.summary.scalar('Base Val Loss Step' + str(j+1), total_lossb[j]))

            self.training_summ_op = tf.summary.merge(self.training_summaries)

            self.input_val_loss = tf.placeholder(tf.float32)
            self.input_val_acc = tf.placeholder(tf.float32)
            self.val_summaries = []
            self.val_summaries.append(tf.summary.scalar('Meta Val Loss', self.input_val_loss))
            self.val_summaries.append(tf.summary.scalar('Meta Val Accuracy', self.input_val_acc))
            self.val_summ_op = tf.summary.merge(self.val_summaries)


        def construct_model_disc(self):
            """The function to construct meta-train model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32) # episode train images
            self.inputb = tf.placeholder(tf.float32) # episode test images
            self.labela = tf.placeholder(tf.float32) # episode train labels
            self.labelb = tf.placeholder(tf.float32) # episode test labels
            self.labeldisc = tf.placeholder(tf.float32) # episode test labels

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()


                # Generate empty list to record losses

                # Embed the input images to embeddings with ss weights
                _,self.emb_outputa = self.forward_resnet_disc(self.inputa, weights, ss_weights,dweights, reuse=tf.AUTO_REUSE) # Embed episode train
                _,self.emb_outputb = self.forward_resnet_disc(self.inputb, weights, ss_weights,dweights, reuse=True) # Embed episode test

                normalize_a = tf.nn.l2_normalize(self.emb_outputa,1)        
                normalize_b = tf.nn.l2_normalize(self.emb_outputb,1)
                self.d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                
                #d = cosine_similarity(emb_outputa,emb_outputb);
                
                self.distance = tf.reshape(self.d,[-1])
                self.lossdisc = tf.reduce_mean(tf.square(self.distance - self.labeldisc))
                
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.metatrain_op = optimizer.minimize(self.lossdisc, var_list=list(dweights.values()))


                
        
        def construct_model_disc_KTNet(self):
            """The function to construct meta-train model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32) # episode train images
            self.inputb = tf.placeholder(tf.float32) # episode test images
            self.b_ssl = tf.placeholder(tf.float32) # episode train images
            self.labela = tf.placeholder(tf.float32) # episode train labels
            self.labelb = tf.placeholder(tf.float32) # episode test labels
            self.labeldisc = tf.placeholder(tf.float32) # episode test labels
            self.label_ssl = tf.placeholder(tf.float32) # episode test labels

            with tf.variable_scope('meta-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()
                self.tr_weights = tr_weights = self.construct_transformer_weights()
                # Load base epoch number from FLAGS
                num_updates = FLAGS.train_base_epoch_num

                def task_metalearn(inp, reuse=True):
                    """The function to process one episode in a meta-batch.
                    Args:
                      inp: the input episode.
                      reuse: whether reuse the variables for the normalization.
                    Returns:
                      A serious outputs like losses and accuracies.
                    """
                    # Seperate inp to different variables
                    
                    
                    inputa, inputb, labela, labelb,ldisc,bssl,lssl = inp
                    #inputa, inputb, labela, labelb,ldisc = inp
                    # Generate empty list to record losses
                    lossa_list = [] # Base train loss list
                    lossb_list = [] # Base test loss list
                    #inputa,bssl,labela,labelb,lssl = inp
                    # Embed the input images to embeddings with ss weights

                    _,self.emb_outputa = self.forward_resnet_disc_ss(inputa, weights, ss_weights,dweights,dss_weights, reuse=tf.AUTO_REUSE) # Embed episode train
                    _,self.emb_outputb = self.forward_resnet_disc_ss(inputb, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test
                    _,self.emb_outputbssl = self.forward_resnet_disc_ss(bssl, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test

                    

################################################################ SSL ########################################################

############################### Calculate SSL loss###################################################
                    
                    outputa = self.forward_transformer(self.emb_outputa, tr_weights,reuse=tf.AUTO_REUSE)
                    outputbssl = self.forward_transformer(self.emb_outputbssl, tr_weights,True)
                    
                    normalize_a = tf.nn.l2_normalize(outputa,1)        
                    normalize_bssl = tf.nn.l2_normalize(outputbssl,1)
                    d = tf.matmul(normalize_a, normalize_bssl, transpose_b=True)
                
                
                    distance = tf.reshape(d,[-1])
                    lossdisc = tf.reduce_mean(tf.square(distance - lssl))


                    lossa = lossdisc
                    # Record base train loss
                    lossa_list.append(lossa)

############################## Calculate Actual Discriminator LOSS#######################################
                    
                    outputb = self.forward_transformer(self.emb_outputb, tr_weights,True)  
                    normalize_b = tf.nn.l2_normalize(outputb,1)
                    d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                    
                    
                    distance = tf.reshape(d,[-1])
                    lossdisc = tf.reduce_mean(tf.square(distance - ldisc))
                    
                    
                    lossb = lossdisc
                    # Record base test loss
                    lossb_list.append(lossb)
                    # Calculate the gradients for the fc layer
                    grads = tf.gradients(lossa, list(tr_weights.values()))
                    gradients = dict(zip(tr_weights.keys(), grads))
                    # Use graient descent to update the fc layer
                    fast_tr_weights = dict(zip(tr_weights.keys(), [tr_weights[key] - \
                        self.update_lr*gradients[key] for key in tr_weights.keys()]))
                    #
                    for j in range(num_updates - 1):
                    #    # Run the following base epochs, these are similar to the first base epoch
                    #
################################ Calculate SSL loss###################################################
                        outputa = self.forward_transformer(self.emb_outputa, fast_tr_weights,True)
                        outputbssl = self.forward_transformer(self.emb_outputbssl, fast_tr_weights,True)
                    #
                        normalize_a = tf.nn.l2_normalize(outputa,1)        
                        normalize_bssl = tf.nn.l2_normalize(outputbssl,1)
                        d = tf.matmul(normalize_a, normalize_bssl, transpose_b=True)
                    #
                    #
                        distance = tf.reshape(d,[-1])
                        lossdisc = tf.reduce_mean(tf.square(distance - lssl))
                    #
                    #
                        lossa = lossdisc
                    #
                    #    
                    #    # Record base train loss
                        lossa_list.append(lossa)
                    #
############################### Calculate Actual Discriminator LOSS#######################################
                        outputb = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)  
                        normalize_b = tf.nn.l2_normalize(outputb,1)
                        d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                    
                    
                        distance = tf.reshape(d,[-1])
                        lossdisc = tf.reduce_mean(tf.square(distance - ldisc))
                    
                    
                        lossb = lossdisc
                        # Record base test loss
                        lossb_list.append(lossb)
                    


                        grads = tf.gradients(lossa, list(fast_tr_weights.values()))
                        gradients = dict(zip(fast_tr_weights.keys(), grads))
                        fast_tr_weights = dict(zip(fast_tr_weights.keys(), [fast_tr_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_tr_weights.keys()]))
                    #    
                    #
                    ## Calculate final episode test predictions
                    outputa = self.forward_transformer(self.emb_outputa, fast_tr_weights,True)  
                    outputb = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)  
                    normalize_a = tf.nn.l2_normalize(outputa,1) 
                    normalize_b = tf.nn.l2_normalize(outputb,1)
                    d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                    
                    
                    resdistance = tf.reshape(d,[-1])
                    lossdisc = tf.reduce_mean(tf.square(resdistance - ldisc))
                    
                    
                    lossb = lossdisc
                    ## Calculate the final episode test loss, it is the loss for the episode on meta-train 
                    #
############################### Calculate Actual Discriminator LOSS#######################################
                    #
                    final_lossb = lossb

                    task_output = [final_lossb, lossb_list, lossa_list,resdistance]
                    

                    return task_output

################################################### END SSL ################################################

                # Initial the batch normalization weights
                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.labeldisc[0],self.b_ssl[0],self.label_ssl[0]), False)
                
                    
                out_dtype = [tf.float32, [tf.float32]*num_updates, [tf.float32]*num_updates,tf.float32]
                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb,self.labeldisc,self.b_ssl,self.label_ssl), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)


                # Seperate the outputs to different variables
                #lossb = result
                lossb, lossesb, lossesa,dis = result


            # Set the variables to output from the tensorflow graph
            self.total_loss = total_loss = lossb / tf.to_float(FLAGS.meta_batch_size)

            self.total_lossa = total_lossa = [tf.reduce_sum(lossesa[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.total_lossb = total_lossb = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            self.similarity = dis

            
            # Set the meta-train optimizer
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.metatrain_op = optimizer.minimize(self.total_loss, var_list=list(dss_weights.values()) + list(tr_weights.values()))





        def construct_test_model_disc(self):
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
            
            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()

                _,self.ea = self.forward_resnet_disc(self.inputa, weights, ss_weights,dweights, reuse=tf.AUTO_REUSE) # Embed episode train
                _,self.eb = self.forward_resnet_disc(self.inputb, weights, ss_weights,dweights, reuse=True) # Embed episode test

                normalize_a = tf.nn.l2_normalize(self.ea,1)        
                normalize_b = tf.nn.l2_normalize(self.eb,1)
                self.d = tf.matmul(normalize_a, normalize_b, transpose_b=True)


        def construct_test(self):
            """The function to construct meta-test model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
            

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
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
                    updatepred = []

                    # Embed the input images to embeddings with ss weights
                    emb_outputa = self.forward_resnet(inputa, self.weights, self.ss_weights, reuse=tf.AUTO_REUSE)
                    emb_outputb = self.forward_resnet(inputb, self.weights, self.ss_weights, reuse=True)

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

                    predicted = tf.nn.softmax(outputb)
                    updatepred.append(predicted)
                    
                    for j in range(num_updates - 1):
                        lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                        grads = tf.gradients(lossa, list(fast_fc_weights.values()))
                        gradients = dict(zip(fast_fc_weights.keys(), grads))
                        fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))
                        outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                        accb_list.append(accb)
                        predictedj = tf.nn.softmax(outputb)
                        updatepred.append(predictedj)

                    lossb = self.loss_func(outputb, labelb)

                    task_output = [lossb, accb, accb_list,updatepred]

                    return task_output

                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates,[tf.float32]*num_updates]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossesb, accsb, accsb_list,updatepred = result
                _,self.ea = self.forward_resnet_disc(self.inputa, weights, ss_weights,dweights, reuse=tf.AUTO_REUSE) # Embed episode train
                _,self.eb = self.forward_resnet_disc(self.inputb, weights, ss_weights,dweights, reuse=True) # Embed episode test

                normalize_a = tf.nn.l2_normalize(self.ea,1)        
                normalize_b = tf.nn.l2_normalize(self.eb,1)
                self.d = tf.matmul(normalize_a, normalize_b, transpose_b=True)



                
            self.updatepredictions = updatepred
            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]





        def construct_test_disc_ktnet(self):
            """The function to construct meta-test model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32) # episode train images
            self.inputb = tf.placeholder(tf.float32) # episode test images
            self.b_ssl = tf.placeholder(tf.float32) # episode train images
            self.labela = tf.placeholder(tf.float32) # episode train labels
            self.labelb = tf.placeholder(tf.float32) # episode test labels
            self.labeldisc = tf.placeholder(tf.float32) # episode test labels
            self.label_ssl = tf.placeholder(tf.float32) # episode test labels
            

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()
                self.tr_weights = tr_weights = self.construct_transformer_weights()

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
                    # Generate empty list to record accuracies
                    accb_list = []
                    updatepred = []

                    inputa, inputb, labela, labelb,ldisc,bssl,lssl = inp
                    #inputa, inputb, labela, labelb,ldisc = inp
                    # Generate empty list to record losses
                    #inputa,bssl,labela,labelb,lssl = inp
                    # Embed the input images to embeddings with ss weights
                    
                    emba,self.emb_outputa = self.forward_resnet_disc_ss(inputa, weights, ss_weights,dweights,dss_weights, reuse=tf.AUTO_REUSE) # Embed episode train
                    embb,self.emb_outputb = self.forward_resnet_disc_ss(inputb, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test
                    _,self.emb_outputbssl = self.forward_resnet_disc_ss(bssl, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test

                    fast_tr_weights = self.tr_weights
################################################################################# DISCRIMINATOR ###########################################################################
                    outputa = self.forward_transformer(self.emb_outputb, tr_weights,reuse=tf.AUTO_REUSE)
                    outputbssl = self.forward_transformer(self.emb_outputbssl, tr_weights,True)
                    
                    normalize_a = tf.nn.l2_normalize(outputa,1)        
                    normalize_bssl = tf.nn.l2_normalize(outputbssl,1)
                    d = tf.matmul(normalize_a, normalize_bssl, transpose_b=True)
                    
                    
                    distance = tf.reshape(d,[-1])
                    lossdisc = tf.reduce_mean(tf.square(distance - lssl))
                    
                    
                    lossa = lossdisc
                    
                    # Record base test loss
                    # Calculate the gradients for the fc layer

                    grads = tf.gradients(lossa, list(tr_weights.values()))
                    gradients = dict(zip(tr_weights.keys(), grads))
                    # Use graient descent to update the fc layer
                    fast_tr_weights = dict(zip(tr_weights.keys(), [tr_weights[key] - \
                        self.update_lr*10*gradients[key] for key in tr_weights.keys()]))

################################################################################# DISCRIMINATOR ###########################################################################
                    # This part is similar to the meta-train function, you may refer to the comments above
                    oa = self.forward_fc(emba, fc_weights)
                    la = self.loss_func(oa, labela)
                    gr = tf.gradients(la, list(fc_weights.values()))
                    grads = dict(zip(fc_weights.keys(), gr))
                    fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                        self.update_lr*grads[key] for key in fc_weights.keys()]))
                    ob = self.forward_fc(embb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(ob), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                    predicted = tf.nn.softmax(ob)
                    updatepred.append(predicted)
                    
                    for j in range(num_updates - 1):
                        la = self.loss_func(self.forward_fc(emba, fast_fc_weights), labela)
                        gr = tf.gradients(la, list(fast_fc_weights.values()))
                        grads = dict(zip(fast_fc_weights.keys(), gr))
                        fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                            self.update_lr*grads[key] for key in fast_fc_weights.keys()]))
                        ob = self.forward_fc(embb, fast_fc_weights)
                        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(ob), 1), tf.argmax(labelb, 1))
                        accb_list.append(accb)
                        predictedj = tf.nn.softmax(ob)
                        updatepred.append(predictedj)



################################################################################# DISCRIMINATOR ###########################################################################
                        ################################ Calculate SSL loss###################################################
                        outputa = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)
                        outputbssl = self.forward_transformer(self.emb_outputbssl, fast_tr_weights,True)
                    #   
                        normalize_a = tf.nn.l2_normalize(outputa,1)        
                        normalize_bssl = tf.nn.l2_normalize(outputbssl,1)
                        d = tf.matmul(normalize_a, normalize_bssl, transpose_b=True)
                    #   
                    #   
                        distance = tf.reshape(d,[-1])
                        lossdisc = tf.reduce_mean(tf.square(distance - lssl))
                    #   
                    #   
                        lossa = lossdisc
                    #   
############################### Calculate Actual Discriminator LOSS#######################################
                        outputb = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)  
                        normalize_b = tf.nn.l2_normalize(outputb,1)
                        d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                        
                        
                        distance = tf.reshape(d,[-1])
                        
                        grads = tf.gradients(lossa, list(fast_tr_weights.values()))
                        gradients = dict(zip(fast_tr_weights.keys(), grads))
                        fast_tr_weights = dict(zip(fast_tr_weights.keys(), [fast_tr_weights[key] - \
                            self.update_lr*10*gradients[key] for key in fast_tr_weights.keys()]))
################################################################################# DISCRIMINATOR ###########################################################################
                    outputa = self.forward_transformer(self.emb_outputa, fast_tr_weights,True)  
                    outputb = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)  
                    normalize_a = tf.nn.l2_normalize(outputa,1) 
                    normalize_b = tf.nn.l2_normalize(outputb,1)
                    d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                    

                    lb = self.loss_func(ob, labelb)

                    task_output = [lb, accb, accb_list,updatepred,d,fast_tr_weights]

                    return task_output

                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.labeldisc[0],self.b_ssl[0],self.label_ssl[0]), False)

                out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates,[tf.float32]*num_updates,tf.float32,{'transformer':tf.float32}]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb,self.labeldisc,self.b_ssl,self.label_ssl), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

                lossesb, accsb, accsb_list,updatepred,dis,tr = result




            
            
            self.updated_tr_weights = tr
            self.d = dis   
            self.updatepredictions = updatepred
            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]



        

        def construct_test_model(self,classes,n):
            """The function to construct meta-test model."""
            # Set the placeholder for the input episode

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.fc_weights = fc_weights = self.construct_fc_weights_O(classes,n)

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
                    updatepred = []

                    # Embed the input images to embeddings with ss weights
                    emb_outputa = self.forward_resnet(inputa, self.weights, self.ss_weights, reuse=tf.AUTO_REUSE)
                    emb_outputb = self.forward_resnet(inputb, self.weights, self.ss_weights, reuse=True)

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

                    predicted = tf.nn.softmax(outputb)
                    updatepred.append(predicted)
                    
                    for j in range(num_updates - 1):
                        lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                        grads = tf.gradients(lossa, list(fast_fc_weights.values()))
                        gradients = dict(zip(fast_fc_weights.keys(), grads))
                        fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                            self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))
                        outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                        accb_list.append(accb)
                        predictedj = tf.nn.softmax(outputb)
                        updatepred.append(predictedj)

                    lossb = self.loss_func(outputb, labelb)

                    task_output = [lossb, accb, accb_list,updatepred]

                    return task_output

                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

                out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates,[tf.float32]*num_updates]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
                lossesb, accsb, accsb_list,updatepred = result




                
            self.updatepredictions = updatepred
            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]




        def construct_test_zero(self):
            """The function to construct meta-test model."""
            # Set the placeholder for the input episode
            self.inputa = tf.placeholder(tf.float32) # episode train images
            self.inputb = tf.placeholder(tf.float32) # episode test images
            self.b_ssl = tf.placeholder(tf.float32) # episode train images
            self.labela = tf.placeholder(tf.float32) # episode train labels
            self.labelb = tf.placeholder(tf.float32) # episode test labels
            self.labeldisc = tf.placeholder(tf.float32) # episode test labels
            self.label_ssl = tf.placeholder(tf.float32) # episode test labels
            

            with tf.variable_scope('meta-test-model', reuse=None) as training_scope:
                # construct the model weights
                self.ss_weights,self.dss_weights = ss_weights,dss_weights = self.construct_resnet_ss_weights()
                self.weights,self.dweights = weights,dweights = self.construct_resnet_weights()
                self.fc_weights = fc_weights = self.construct_fc_weights()
                self.tr_weights = tr_weights = self.construct_transformer_weights()

                # Load test base epoch number from FLAGS
                num_updates = FLAGS.test_base_epoch_num
                #num_updates = 2

                def task_metalearn(inp, reuse=True):
                    """The function to process one episode in a meta-batch.
                    Args:
                      inp: the input episode.
                      reuse: whether reuse the variables for the normalization.
                    Returns:
                      A serious outputs like losses and accuracies.
                    """
                    # Seperate inp to different variables
                    # Generate empty list to record accuracies
                    accb_list = []
                    updatepred = []

                    inputa, inputb, labela, labelb,ldisc,bssl,lssl = inp
                    emba,self.emb_outputa = self.forward_resnet_disc_ss(inputa, weights, ss_weights,dweights,dss_weights, reuse=tf.AUTO_REUSE) # Embed episode train
                    embb,self.emb_outputb = self.forward_resnet_disc_ss(inputb, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test
                    _,self.emb_outputbssl = self.forward_resnet_disc_ss(bssl, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test

                    fast_tr_weights = self.tr_weights
################################################################################# DISCRIMINATOR ###########################################################################
                    outputa = self.forward_transformer(self.emb_outputb, tr_weights,reuse=tf.AUTO_REUSE)
                    outputbssl = self.forward_transformer(self.emb_outputbssl, tr_weights,True)
                    
                    normalize_a = tf.nn.l2_normalize(outputa,1)        
                    normalize_bssl = tf.nn.l2_normalize(outputbssl,1)
                    d = tf.matmul(normalize_a, normalize_bssl, transpose_b=True)
                    
                    
                    distance = tf.reshape(d,[-1])
                    lossdisc = tf.reduce_mean(tf.square(distance - lssl))
                    
                    
                    lossa = lossdisc
                    
                    # Record base test loss
                    # Calculate the gradients for the fc layer

                    grads = tf.gradients(lossa, list(tr_weights.values()))
                    gradients = dict(zip(tr_weights.keys(), grads))
                    # Use graient descent to update the fc layer
                    fast_tr_weights = dict(zip(tr_weights.keys(), [tr_weights[key] - \
                        self.update_lr*10*gradients[key] for key in tr_weights.keys()]))

################################################################################# DISCRIMINATOR ###########################################################################
                    # This part is similar to the meta-train function, you may refer to the comments above
                    oa = self.forward_fc(emba, fc_weights)
                    la = self.loss_func(oa, labela)
                    gr = tf.gradients(la, list(fc_weights.values()))
                    grads = dict(zip(fc_weights.keys(), gr))
                    fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                        self.update_lr*grads[key] for key in fc_weights.keys()]))
                    ob = self.forward_fc(embb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(ob), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                    predicted = tf.nn.softmax(ob)
                    updatepred.append(predicted)
                    
                    for j in range(num_updates - 1):
                        la = self.loss_func(self.forward_fc(emba, fast_fc_weights), labela)
                        gr = tf.gradients(la, list(fast_fc_weights.values()))
                        grads = dict(zip(fast_fc_weights.keys(), gr))
                        fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                            self.update_lr*grads[key] for key in fast_fc_weights.keys()]))
                        ob = self.forward_fc(embb, fast_fc_weights)
                        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(ob), 1), tf.argmax(labelb, 1))
                        accb_list.append(accb)
                        predictedj = tf.nn.softmax(ob)
                        updatepred.append(predictedj)



################################################################################# DISCRIMINATOR ###########################################################################
                        ################################ Calculate SSL loss###################################################
                        outputa = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)
                        outputbssl = self.forward_transformer(self.emb_outputbssl, fast_tr_weights,True)
                    #   
                        normalize_a = tf.nn.l2_normalize(outputa,1)        
                        normalize_bssl = tf.nn.l2_normalize(outputbssl,1)
                        d = tf.matmul(normalize_a, normalize_bssl, transpose_b=True)
                    #   
                    #   
                        distance = tf.reshape(d,[-1])
                        lossdisc = tf.reduce_mean(tf.square(distance - lssl))
                    #   
                    #   
                        lossa = lossdisc
                    #   
############################### Calculate Actual Discriminator LOSS#######################################
                        outputb = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)  
                        normalize_b = tf.nn.l2_normalize(outputb,1)
                        d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                        
                        
                        distance = tf.reshape(d,[-1])
                        
                        grads = tf.gradients(lossa, list(fast_tr_weights.values()))
                        gradients = dict(zip(fast_tr_weights.keys(), grads))
                        fast_tr_weights = dict(zip(fast_tr_weights.keys(), [fast_tr_weights[key] - \
                            self.update_lr*10*gradients[key] for key in fast_tr_weights.keys()]))
################################################################################# DISCRIMINATOR ###########################################################################
                    outputa = self.forward_transformer(self.emb_outputa, fast_tr_weights,True)  
                    outputb = self.forward_transformer(self.emb_outputb, fast_tr_weights,True)  
                    normalize_a = tf.nn.l2_normalize(outputa,1) 
                    normalize_b = tf.nn.l2_normalize(outputb,1)
                    d = tf.matmul(normalize_a, normalize_b, transpose_b=True)
                    

                    lb = self.loss_func(ob, labelb)

                    task_output = [lb, accb, accb_list,updatepred,d,fast_tr_weights]

                    return task_output

                if FLAGS.norm is not None:
                    unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.labeldisc[0],self.b_ssl[0],self.label_ssl[0]), False)

                out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates,[tf.float32]*num_updates,tf.float32,{'transformer':tf.float32}]

                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb,self.labeldisc,self.b_ssl,self.label_ssl), \
                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

                lossesb, accsb, accsb_list,updatepred,dis,tr = result


                
            emba,ea = self.forward_resnet_disc_ss(self.inputa, weights, ss_weights,dweights,dss_weights, reuse=tf.AUTO_REUSE) # Embed episode train
            embb,eb = self.forward_resnet_disc_ss(self.inputb, weights, ss_weights,dweights,dss_weights, reuse=True) # Embed episode test

            outa = self.forward_transformer(ea, self.tr_weights,True)  
            outb = self.forward_transformer(eb, self.tr_weights,True)  
            norm_a = tf.nn.l2_normalize(outa,1) 
            norm_b = tf.nn.l2_normalize(outb,1)
            self.sim = tf.matmul(norm_a, norm_b, transpose_b=True)
            
            
            self.updated_tr_weights = tr
            self.d = dis   
            self.updatepredictions = updatepred
            self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
            self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
            self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]

        

    return MetaModel()
