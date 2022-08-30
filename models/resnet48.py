
""" ResNet-12 class. """
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block,conv,SQ,resnet_conv_block_wb

FLAGS = flags.FLAGS

class Models:
    """The class that contains the code for the basic resnet models and SS weights"""
    def __init__(self):
        # Set the dimension number for the input feature maps
        self.dim_input = FLAGS.img_size * FLAGS.img_size * 3
        # Set the dimension number for the outputs
        self.dim_output = FLAGS.way_num
        # Load base learning rates from FLAGS
        self.update_lr = FLAGS.base_lr
        # Load the pre-train phase class number from FLAGS
        self.pretrain_class_num = FLAGS.pretrain_class_num
        # Set the initial meta learning rate
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        # Set the initial pre-train learning rate
        self.pretrain_lr = tf.placeholder_with_default(FLAGS.pre_lr, ())

        # Set the default objective functions for meta-train and pre-train
        self.loss_func = xent
        self.pretrain_loss_func = softmaxloss

        # Set the default channel number to 3
        self.channels = 3
        # Load the image size from FLAGS
        self.img_size = FLAGS.img_size

    def process_ss_weights(self, weights, ss_weights, label):
        """The function to process the scaling operation
        Args:
          weights: the weights for the resnet.
          ss_weights: the weights for scaling and shifting operation.
          label: the label to indicate which layer we are operating.
        Return:
          The processed weights for the new resnet.
        """  
        [dim0, dim1] = weights[label].get_shape().as_list()[0:2]
        this_ss_weights = tf.tile(ss_weights[label], multiples=[dim0, dim1, 1, 1])
        return tf.multiply(weights[label], this_ss_weights)

    def forward_pretrain_resnet(self, inp, weights, reuse=False, scope=''):
        """The function to forward the resnet during pre-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        net = self.pretrain_block_forward(inp, weights, 'block1', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block2_a', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block2', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block3_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block3_b', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block3', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block4_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_b', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_c', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block4', reuse, scope)
        net = tf.nn.avg_pool(net, [1,5,5,1], [1,5,5,1], 'VALID')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net

    def forward_resnet(self, inp, weights, ss_weights, reuse=False, scope=''):
        """The function to forward the resnet during meta-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          ss_weights: input scaling weights.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        net = self.block_forward(inp, weights, ss_weights, 'block1', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block2_a', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block2', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block3_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block3_b', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block3', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block4_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_b', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_c', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block4', reuse, scope)
        net = tf.nn.avg_pool(net, [1,5,5,1], [1,5,5,1], 'VALID')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net


    def forward_resnet_disc(self, inp, weights, ss_weights, dweights, reuse=False, scope=''):
        """The function to forward the resnet during meta-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          ss_weights: input scaling weights.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        net = self.block_forward(inp, weights, ss_weights, 'block1', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block2_a', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block2', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block3_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block3_b', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block3', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block4_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_b', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_c', reuse, scope)
        netc = self.block_forward(net, weights, ss_weights, 'block4', reuse, scope)
        
        netd = self.pretrain_block_forward_same(netc, dweights, 'block5_a', reuse, scope)
        netd = self.pretrain_block_forward_same(netd, dweights, 'block5_b', reuse, scope)
        netd = self.pretrain_block_forward_same(netd, dweights, 'block5_c', reuse, scope)
        netd = self.pretrain_block_forward(netd, dweights, 'block5', reuse, scope)
        #netd = self.block_forward(netd, dweights, dss_weights, 'block5', reuse, scope)
        netd = tf.nn.avg_pool(netd, [1,2,2,1], [1,2,2,1], 'VALID')
        netd = tf.reshape(netd, [-1, np.prod([int(dim) for dim in netd.get_shape()[1:]])])

        net = tf.nn.avg_pool(netc, [1,5,5,1], [1,5,5,1], 'VALID')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net,netd


    def forward_resnet_disc_ss(self, inp, weights, ss_weights, dweights,dss_weights, reuse=False, scope=''):
        """The function to forward the resnet during meta-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          ss_weights: input scaling weights.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        net = self.block_forward(inp, weights, ss_weights, 'block1', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block2_a', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block2', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block3_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block3_b', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block3', reuse, scope)

        net = self.pretrain_block_forward_same(net, weights, 'block4_a', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_b', reuse, scope)
        net = self.pretrain_block_forward_same(net, weights, 'block4_c', reuse, scope)
        netc = self.block_forward(net, weights, ss_weights, 'block4', reuse, scope)
        
        netd = self.pretrain_block_forward_same(netc, dweights, 'block5_a', reuse, scope)
        netd = self.pretrain_block_forward_same(netd, dweights, 'block5_b', reuse, scope)
        netd = self.pretrain_block_forward_same(netd, dweights, 'block5_c', reuse, scope)
        #netd = self.pretrain_block_forward(netd, dweights, 'block5', reuse, scope)
        netd = self.block_forward(netd, dweights, dss_weights, 'block5', reuse, scope)
        #netd = tf.nn.avg_pool(netd, [1,2,2,1], [1,2,2,1], 'VALID')
        #netd = tf.reshape(netd, [-1, np.prod([int(dim) for dim in netd.get_shape()[1:]])])

        net = tf.nn.avg_pool(netc, [1,5,5,1], [1,5,5,1], 'VALID')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net,netd


    def forward_transformer(self, inp, tr_weights,reuse):
        """The function to forward the fc layer
        Args:
          inp: input feature maps.
          fc_weights: input fc weights.
        Return:
          The processed feature maps.
        """  
        res = resnet_nob_conv_block(inp, tr_weights['transformer'], reuse, 'transformer')
        res = tf.nn.avg_pool(res, [1,2,2,1], [1,2,2,1], 'VALID')
        res = tf.reshape(res, [-1, np.prod([int(dim) for dim in res.get_shape()[1:]])])
        return res

    def target_category_loss(x, category_index, nb_classes):
        return tf.multiply(x, tf.one_hot([category_index], nb_classes))


    def forward_fc(self, inp, fc_weights):
        """The function to forward the fc layer
        Args:
          inp: input feature maps.
          fc_weights: input fc weights.
        Return:
          The processed feature maps.
        """  
        net = tf.matmul(inp, fc_weights['w5']) + fc_weights['b5']
        return net

    def pretrain_block_forward(self, inp, weights, block, reuse, scope):
        """The function to forward a resnet block during pre-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          block: the string to indicate which block we are processing.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        net = resnet_conv_block(inp, weights[block + '_conv1'], weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, weights[block + '_conv2'], weights[block + '_bias2'], reuse, scope+block+'1')
        net = resnet_conv_block(net, weights[block + '_conv3'], weights[block + '_bias3'], reuse, scope+block+'2')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], 'VALID')
        net = tf.nn.dropout(net, keep_prob=FLAGS.pretrain_dropout_keep)
        return net

    def pretrain_block_forward_same(self, inp, weights, block, reuse, scope):
        """The function to forward a resnet block during pre-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          block: the string to indicate which block we are processing.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        net = resnet_conv_block(inp, weights[block + '_conv1'], weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, weights[block + '_conv2'], weights[block + '_bias2'], reuse, scope+block+'1')
        net = resnet_conv_block(net, weights[block + '_conv3'], weights[block + '_bias3'], reuse, scope+block+'2')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        net = tf.nn.dropout(net, keep_prob=FLAGS.pretrain_dropout_keep)
        return net

    def block_forward(self, inp, weights, ss_weights, block, reuse, scope):
        """The function to forward a resnet block during meta-train phase
        Args:
          inp: input feature maps.
          weights: input resnet weights.
          ss_weights: input scaling weights.
          block: the string to indicate which block we are processing.
          reuse: reuse the batch norm weights or not.
          scope: the label to indicate which layer we are processing.
        Return:
          The processed feature maps.
        """  
        net = resnet_conv_block(inp, self.process_ss_weights(weights, ss_weights, block + '_conv1'), \
            ss_weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, self.process_ss_weights(weights, ss_weights, block + '_conv2'), \
            ss_weights[block + '_bias2'], reuse, scope+block+'1')
        net = resnet_conv_block(net, self.process_ss_weights(weights, ss_weights, block + '_conv3'), \
            ss_weights[block + '_bias3'], reuse, scope+block+'2')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], 'VALID')
        net = tf.nn.dropout(net, keep_prob=1)
        return net

    def construct_fc_weights_O(self,output,n):
        """The function to construct fc weights.
        Return:
          The fc weights.
        """  

        with tf.variable_scope("block0"+str(output)+str(n), reuse=False) as scope:
        
            dtype = tf.float32
            fc_weights = {}
            fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
            if FLAGS.phase=='pre':
                fc_weights['w5'] = tf.get_variable('fc_w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
                fc_weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='fc_b5')
            else:
                fc_weights['w5'] = tf.get_variable('fc_w5', [512, output], initializer=fc_initializer)
                fc_weights['b5'] = tf.Variable(tf.zeros([output]), name='fc_b5')
            return fc_weights

    def construct_fc_weights(self):
        """The function to construct fc weights.
        Return:
          The fc weights.
        """  
        dtype = tf.float32
        fc_weights = {}
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        if FLAGS.phase=='pre':
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='fc_b5')
        else:
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, self.dim_output], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='fc_b5')
        return fc_weights

    def construct_resnet_weights(self):
        """The function to construct resnet weights.
        Return:
          The resnet weights.
        """  
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        weights = self.construct_residual_block_weights(weights, 3, 3, 64, conv_initializer, dtype, 'block1')

        weights = self.construct_residual_block_weights(weights, 3, 64, 128, conv_initializer, dtype, 'block2_a')
        weights = self.construct_residual_block_weights(weights, 3, 128, 128, conv_initializer, dtype, 'block2')
        #weights = self.construct_residual_block_weights(weights, 3, 64, 128, conv_initializer, dtype, 'block2')

        weights = self.construct_residual_block_weights(weights, 3, 128, 256, conv_initializer, dtype, 'block3_a')
        weights = self.construct_residual_block_weights(weights, 3, 256, 256, conv_initializer, dtype, 'block3_b')
        weights = self.construct_residual_block_weights(weights, 3, 256, 256, conv_initializer, dtype, 'block3')
        #weights = self.construct_residual_block_weights(weights, 3, 128, 256, conv_initializer, dtype, 'block3')

        weights = self.construct_residual_block_weights(weights, 3, 256, 512, conv_initializer, dtype, 'block4_a')
        weights = self.construct_residual_block_weights(weights, 3, 512, 512, conv_initializer, dtype, 'block4_b')
        weights = self.construct_residual_block_weights(weights, 3, 512, 512, conv_initializer, dtype, 'block4_c')
        weights = self.construct_residual_block_weights(weights, 3, 512, 512, conv_initializer, dtype, 'block4')
        #weights = self.construct_residual_block_weights(weights, 3, 256, 512, conv_initializer, dtype, 'block4')
        
        dweights = {}
        dweights = self.construct_residual_block_weights(dweights, 3, 256, 512, conv_initializer, dtype, 'block5_a')
        dweights = self.construct_residual_block_weights(dweights, 3, 512, 512, conv_initializer, dtype, 'block5_b')
        dweights = self.construct_residual_block_weights(dweights, 3, 512, 512, conv_initializer, dtype, 'block5_c')
        dweights = self.construct_residual_block_weights(dweights, 3, 512, 512, conv_initializer, dtype, 'block5')

        
        with tf.variable_scope("block0", reuse=tf.AUTO_REUSE) as scope:

            weights['w5'] = tf.get_variable('w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='b5')
        return weights,dweights



    def construct_residual_block_weights(self, weights, k, last_dim_hidden, dim_hidden, conv_initializer, dtype, scope='block0'):
        """The function to construct one block of the resnet weights.
        Args:
          weights: the resnet weight list.
          k: the dimension number of the convolution kernel.
          last_dim_hidden: the hidden dimension number of last block.
          dim_hidden: the hidden dimension number of the block.
          conv_initializer: the convolution initializer.
          dtype: the dtype for numpy.
          scope: the label to indicate which block we are processing.
        Return:
          The resnet block weights.
        """ 
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as s:
        
            weights[scope + '_conv1'] = tf.get_variable(scope + '_conv1', [k, k, last_dim_hidden, dim_hidden], \
                initializer=conv_initializer, dtype=dtype)
            weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
            weights[scope + '_conv2'] = tf.get_variable(scope + '_conv2', [k, k, dim_hidden, dim_hidden], \
                initializer=conv_initializer, dtype=dtype)
            weights[scope + '_bias2'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias2')
            weights[scope + '_conv3'] = tf.get_variable(scope + '_conv3', [k, k, dim_hidden, dim_hidden], \
                initializer=conv_initializer, dtype=dtype)
            weights[scope + '_bias3'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias3')
            weights[scope + '_conv_res'] = tf.get_variable(scope + '_conv_res', [1, 1, last_dim_hidden, dim_hidden], \
                initializer=conv_initializer, dtype=dtype)
        return weights

    def construct_transformer_weights(self):
        """The function to construct fc weights.
        Return:
          The fc weights.
        """  
        dtype = tf.float32
        tr_weights = {}
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        tr_weights['transformer'] = tf.get_variable('transformer', [1, 1, 512, 512], \
                initializer=conv_initializer, dtype=dtype)
        return tr_weights

    def construct_resnet_ss_weights(self):
        """The function to construct ss weights.
        Return:
          The ss weights.
        """ 
        ss_weights = {}
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 3, 64, 'block1')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 128, 128, 'block2')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 256, 256, 'block3')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 512, 512, 'block4')
        dss_weights = {}
        dss_weights = self.construct_residual_block_ss_weights(dss_weights, 512, 512, 'block5')
        return ss_weights,dss_weights

    def construct_residual_block_ss_weights(self, ss_weights, last_dim_hidden, dim_hidden, scope='block0'):
        """The function to construct one block ss weights.
        Args:
          ss_weights: the ss weight list.
          last_dim_hidden: the hidden dimension number of last block.
          dim_hidden: the hidden dimension number of the block.
          scope: the label to indicate which block we are processing.
        Return:
          The ss block weights.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as s:
            ss_weights[scope + '_conv1'] = tf.Variable(tf.ones([1, 1, last_dim_hidden, dim_hidden]), name=scope + '_conv1')
            ss_weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
            ss_weights[scope + '_conv2'] = tf.Variable(tf.ones([1, 1, dim_hidden, dim_hidden]), name=scope + '_conv2')
            ss_weights[scope + '_bias2'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias2')
            ss_weights[scope + '_conv3'] = tf.Variable(tf.ones([1, 1, dim_hidden, dim_hidden]), name=scope + '_conv3')
            ss_weights[scope + '_bias3'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias3')
        return ss_weights


        

