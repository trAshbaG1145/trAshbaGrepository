from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # TODO: 初始化两层网络的权重和偏差。权重应该从一个以0.0为中心的高斯函数初始化，
        # 标准差等于weight_scale，偏差应该初始化为0。所有的权重和偏差都应该存储在字典self中。
        # 参数，第一层权重和偏差使用键'W1'和'b1'，第二层权重和偏差使用键'W2'和'b2'。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 第一层: W1 (D, H), b1 (H,)
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        
        # 第二层: W2 (H, C), b2 (C,)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        # TODO: 实现两层网络的正向传递，计算对X进行分类的得分，并将其存储在scores变量中。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 架构: affine - relu - affine - softmax
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # 第一层 (affine - relu)
        # out1 是 ReLU 激活后的输出
        out1, cache1 = affine_relu_forward(X, W1, b1)
        
        # 第二层 (affine)
        # scores 是最终的分类分数
        scores, cache2 = affine_forward(out1, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # TODO: 实现两层网络的向后传递。将损失存储在损失变量中，梯度存储在梯度字典中。
        #  使用softmax计算数据损失，并确保gradients[k]包含self.params[k]的梯度。别忘了添加L2正则化!
        # 注意: 为了确保你的实现与我们的匹配，并通过自动化测试，请确保你的L2正则化包含一个0.5
        # 的因子来简化梯度的表达式。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 1. 计算损失 (Softmax 损失 + L2 正则化)
        loss, dscores = softmax_loss(scores, y)
        # 添加 L2 正则化损失
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # 2. 计算梯度 (反向传播)
        # grads['W2'], grads['b2']
        # 第二层反向传播 (affine)
        dout1, dW2, db2 = affine_backward(dscores, cache2)
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        
        # grads['W1'], grads['b1']
        # 第一层反向传播 (affine - relu)
        dx, dW1, db1 = affine_relu_backward(dout1, cache1)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #

        # TODO: 初始化网络的参数，将所有值存储在self.params字典中。
        # 将第一层的权重和偏差存储在W1和b1中；对于第二层，使用W2和b2等。
        # 权重应从以0为中心的正态分布初始化，标准偏差等于weight_scale。偏差应初始化为零。
        # 当使用批量归一化时，将第一层的缩放和偏移参数存储在gamma1和beta1中；
        # 对于第二层，使用gamma2和beta2等。缩放参数应初始化为1，移位参数应初始化至0。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # layer dimensions: input -> hidden_dims... -> num_classes
        layer_dims = [input_dim] + list(hidden_dims) + [num_classes]

        for i in range(1, self.num_layers + 1):
            W_name = 'W%d' % i
            b_name = 'b%d' % i
            self.params[W_name] = weight_scale * np.random.randn(layer_dims[i - 1], layer_dims[i])
            self.params[b_name] = np.zeros(layer_dims[i])

        # initialize gamma and beta for normalization (for layers 1..L-1)
        if self.normalization is not None:
            for i in range(1, self.num_layers):
                gamma_name = 'gamma%d' % i
                beta_name = 'beta%d' % i
                self.params[gamma_name] = np.ones(layer_dims[i])
                self.params[beta_name] = np.zeros(layer_dims[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #

        # TODO: 实现全连接网络的正向传递，计算X的类分数并将其存储在分数变量中。
        # 当使用dropout时，您需要将self.dropout_param传递给每个dropout正向传递。
        # 使用批处理规范化时，需要将self.bn_params[0]传递到第一个批处理规范层的正向过程，
        # 将self.bn_params[1]传递到第二个批处理标准化层的正向通道，等等。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = []
        out = X

        # Forward for first (L-1) layers
        for i in range(1, self.num_layers):
            W = self.params['W%d' % i]
            b = self.params['b%d' % i]

            # affine
            out, fc_cache = affine_forward(out, W, b)

            # normalization (batchnorm or layernorm)
            bn_cache = None
            if self.normalization == 'batchnorm':
                gamma = self.params['gamma%d' % i]
                beta = self.params['beta%d' % i]
                out, bn_cache = batchnorm_forward(out, gamma, beta, self.bn_params[i - 1])
            elif self.normalization == 'layernorm':
                gamma = self.params['gamma%d' % i]
                beta = self.params['beta%d' % i]
                out, bn_cache = layernorm_forward(out, gamma, beta, self.bn_params[i - 1])

            # relu
            out, relu_cache = relu_forward(out)

            # dropout
            drop_cache = None
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)

            caches.append((fc_cache, bn_cache, relu_cache, drop_cache))

        # Last layer (affine only)
        W_last = self.params['W%d' % self.num_layers]
        b_last = self.params['b%d' % self.num_layers]
        scores, last_cache = affine_forward(out, W_last, b_last)

        # store caches for backward
        self._caches = caches
        self._last_cache = last_cache

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #

        # TODO: 为完全连接的网络实现向后传递。将loss存储在loss variable中，将梯度存储在梯度字典中。
        # 使用softmax计算数据损失，并确保grades[k]保持self.params[k]的梯度。
        # 不要忘记添加L2正则化！
        # 使用批处理 / 层规范化时，不需要正则化缩放和偏移参数。
        # 注意：为了确保您的实现与我们的匹配并通过自动测试，请确保L2正则化包含0.5的因子，
        # 以简化梯度的表达式。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # compute softmax loss
        loss, dscores = softmax_loss(scores, y)

        # add L2 regularization for all weights
        for i in range(1, self.num_layers + 1):
            W = self.params['W%d' % i]
            loss += 0.5 * self.reg * np.sum(W * W)

        # backward into last affine layer
        dout, dW, db = affine_backward(dscores, self._last_cache)
        grads['W%d' % self.num_layers] = dW + self.reg * self.params['W%d' % self.num_layers]
        grads['b%d' % self.num_layers] = db

        # backward for previous layers
        for i in range(self.num_layers - 1, 0, -1):
            fc_cache, bn_cache, relu_cache, drop_cache = self._caches[i - 1]

            # dropout backward
            if self.use_dropout and drop_cache is not None:
                dout = dropout_backward(dout, drop_cache)

            # relu backward
            dout = relu_backward(dout, relu_cache)

            # normalization backward
            if self.normalization == 'batchnorm' and bn_cache is not None:
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                grads['gamma%d' % i] = dgamma
                grads['beta%d' % i] = dbeta
            elif self.normalization == 'layernorm' and bn_cache is not None:
                dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)
                grads['gamma%d' % i] = dgamma
                grads['beta%d' % i] = dbeta

            # affine backward
            dx, dW, db = affine_backward(dout, fc_cache)
            grads['W%d' % i] = dW + self.reg * self.params['W%d' % i]
            grads['b%d' % i] = db

            dout = dx

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads