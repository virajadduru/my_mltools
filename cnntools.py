# -*- coding: utf-8 -*-
#%% imports
import theano
import theano.tensor as T
import cPickle
import sys
import numpy as np
from theano.tensor.signal import downsample
from math import ceil

#%% base classes 

# convolution layer class
class Conv2d(object):
    """
    input_img : 4d tensor
    
    image_shape : [batchsize,num_channels,imagerows,imagecols]
    
    filter_shape : [layersize,num_channels,filterrows,filtercols]
    
    rng : random number generator
    
    strides = [1,1] : stride in each dimension for the filters
    
    sparse_count = 0 : filter sparsity
    
    activation = T.tanh : activation function on the convolution output.
    """
    def __init__(self,input_img,input_shape,
                 filter_shape,rng,strides=[1,1],
                 sparse_count=0,activation=T.tanh,pooling=None):
        
        # num_channels of input must match that of filters        
        assert input_shape[1]==filter_shape[1]
        # checking if image size is smaller than a filter
        assert np.prod(input_shape[2:])>=np.prod(filter_shape[2:])
       
        # making a mask and recalculating filter_shape for sparse_count
        oneZeros = np.append(1,np.zeros(sparse_count))
        x = np.append(np.tile(oneZeros,filter_shape[2]-1),1)
        y = np.append(np.tile(oneZeros,filter_shape[3]-1),1)
        self.mask = np.asarray(np.outer(x,y),dtype=theano.config.floatX) # no need for reshaping for 2d
        filter_shape = filter_shape[0:2]+list(self.mask.shape)
        
        # initializing parameters
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0]*np.prod(filter_shape[2:])
        W_bound = np.sqrt(6.0/(fan_in+fan_out))
        W_values = np.asarray(rng.uniform(low=-W_bound,high=W_bound,
                                          size=filter_shape)*self.mask,
                                          dtype=theano.config.floatX)                          
        self.W = theano.shared(value=W_values,name='W') 
        
        b_values = np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b = theano.shared(value = b_values,name='b')
        
        # perform 2d convolution
        conv_out = T.nnet.conv2d(input=input_img,
                                 filters=self.W,
                                 image_shape = input_shape,
                                 filter_shape = filter_shape,
                                 border_mode ='valid',
                                 subsample=strides)
        # Calculate output size                         
        self.output_shape = [input_shape[0],filter_shape[0]]+\
                            [input_shape[i]-filter_shape[i]+1\
                             for i in range(2,len(input_shape))]
        
        if pooling!=None:
            if len(pooling)!=2:
                raise NotImplementedError()
                
            self.output_shape = self.output_shape[0:2]+\
                        [int(ceil((self.output_shape[i+2]-\
                        pooling[i]+1)/float(pooling[i])))\
                         for i in range(len(pooling))]
            
            conv_out = downsample.max_pool_2d(input=conv_out,
                                  ds=pooling,
                                  ignore_border=True)             
            
            pass                         
                                
        # adding bias                          
        conv_out += self.b.dimshuffle('x',0,'x','x')
        
        # applying activation
        self.output = (conv_out if activation is None
                       else activation(conv_out))

        
        # put params in a list        
        self.params=[self.W,self.b]
        self.masks=[self.mask,[]]


class avg_pool_2d(object):
    """
    input_img : 4d tensor
    
    image_shape : [batchsize,num_channels,imagerows,imagecols]
    
    pooling : [layersize,num_channels,filterrows,filtercols]
    
    activation = T.tanh : activation function on the convolution output.
    """
    def __init__(self,input_img,input_shape,
                 pooling=[2,2],
                 activation=T.tanh):
        
        from math import ceil
        
        filter_shape=[1,1]+pooling
        # checking if image size is smaller than a filter
        assert np.prod(input_shape[2:])>=np.prod(filter_shape[2:])
        
        # initializing parameters
        W_values = np.ones(filter_shape,dtype=theano.config.floatX)   
        self.W = theano.shared(value=W_values,name='W') 
        
        # Calculate output size    
        self.output_shape = input_shape[0:2]+\
                        [int(ceil((input_shape[i+2]-pooling[i]+1)/float(pooling[i])))\
                         for i in range(len(pooling))]
        
        # calculated reshaped input size         
        ip_shape = [np.prod(input_shape[0:2]),1]+input_shape[2:]
        # perform 2d convolution
        conv_out = T.nnet.conv2d(input=input_img.reshape(ip_shape),
                                 filters=self.W,
                                 image_shape=ip_shape,
                                 filter_shape=filter_shape,
                                 border_mode='valid',
                                 subsample=pooling).reshape(self.output_shape)
        
        # applying activation
        self.output = (conv_out if activation is None
                       else activation(conv_out))

        # put params in a list        
        self.params=[]
        self.masks=[]

# Fully connected layer class
class FC(object):
    
    def __init__(self,input_array,n_in,n_out,rng,
                 activation=T.tanh):

        # Create members W and b
        # random initialization of parameter values. needs rng
        W_bound = np.sqrt(6./(n_in+n_out))
        W_values = np.asarray(rng.uniform(low=-W_bound,
                                          high=W_bound,
                                          size=[n_in,n_out]),
                              dtype=theano.config.floatX)                  

        self.W = theano.shared(value=W_values,name='W',borrow=True)
        
        # Creating bias parameters
        b_values = np.zeros((n_out,),dtype=theano.config.floatX)
        self.b = theano.shared(value = b_values,name='b')
        
        # formula for computing output
        output = T.dot(input_array, self.W) + self.b
        self.output = (
            output if activation is None
            else activation(output)
        )
            
        
        # model parameters
        self.params = [self.W,self.b]
        self.masks=[[],[]]
   
   
# needs testing
class max_pool_2d(object):
    """
    input_img : 4d tensor
    
    image_shape : [batchsize,num_channels,imagerows,imagecols]
    
    pooling : [layersize,num_channels,filterrows,filtercols]
    
    activation = T.tanh : activation function on the convolution output.
    """
    def __init__(self,input_img,input_shape,
                 pooling=[2,2],
                 activation=T.tanh):
        
        from math import ceil
        
        
        # Calculate output size    
        self.output_shape = input_shape[0:2]+\
                        [int(ceil((input_shape[i+2]-pooling[i]+1)/float(pooling[i])))\
                         for i in range(len(pooling))]
        
        pool_out = downsample.max_pool_2d(input=input_img,
                                  ds=pooling,
                                  ignore_border=True)
        
        # applying activation
        self.output = (pool_out if activation is None
                       else activation(pool_out))

        # put params in a list        
        self.params=[]
        self.masks=[]

#%% Convolution network class

class ConvNet(object):

    # Parameter loading must be done using load params function

    def __init__(self,input_shape,layersizes,filter_size,rng=np.random.RandomState(234)):
        
        self.x = T.matrix('x',dtype=theano.config.floatX)
        # input needs to be a 4d tensor otherwise conv2d wont accept
        layer0input = self.x.reshape(input_shape)

        self.batchsize = input_shape[0]
        
        self.layer0 = Conv2d(input_img=layer0input,
                             input_shape=input_shape,
                             filter_shape=[layersizes[0],
                                           input_shape[1],
                                           filter_size[0],
                                           filter_size[0]],
                             rng = rng,
                             activation=T.tanh,
                             pooling=[2,2])
                             
        curr_shape = self.layer0.output_shape
        print curr_shape
#        self.layer0pool = max_pool_2d(input_img=self.layer0.output,
#                                      input_shape = curr_shape,
#                                      pooling=[2,2],
#                                      activation=T.tanh)
#                                      
#        curr_shape = self.layer0pool.output_shape                              
        self.layer1 = Conv2d(input_img=self.layer0.output,
                             input_shape=curr_shape,
                             filter_shape=[layersizes[1],
                                           curr_shape[1],
                                           filter_size[1],
                                           filter_size[1]],
                             rng=rng,
                             activation=T.tanh,
                             pooling=[2,2])
                             
        

        curr_shape = self.layer1.output_shape
        print curr_shape
#        self.layer1pool = max_pool_2d(input_img=self.layer1.output,
#                                      input_shape=curr_shape,
#                                      pooling=[2,2],
#                                      activation=T.tanh)
#
#        curr_shape = self.layer1pool.output_shape           
        self.layer2 = FC(input_array=self.layer1.output.flatten(2),
                         n_in=np.prod(curr_shape[1:]),
                         n_out=layersizes[2],
                         rng=rng,
                         activation=T.tanh)
                         
        self.layer3 = FC(input_array=self.layer2.output,
                         n_in=layersizes[2],
                         n_out=layersizes[3],
                         rng=rng,
                         activation=T.nnet.softmax)

                         
        self.py_given_x = self.layer3.output
                         
        self.output= T.argmax(self.py_given_x,axis=1)    
        
        # flattening out the params 
        self.params= self.layer0.params+self.layer1.params+\
                     self.layer2.params+self.layer3.params
        
        self.masks = self.layer0.masks+self.layer1.masks+\
                     self.layer2.masks+self.layer3.masks
        
        self.Ws = [self.layer0.W,self.layer1.W,self.layer2.W,self.layer3.W]

    
    def saveParams(self,file_name):
        try:
            f = file(file_name,'wb')
            for param in self.params:
                cPickle.dump(param.get_value(borrow=True),f)
            f.close()
            print 'params file saved as: '+file_name
        except:
            e = sys.exc_info()[0]
            print( "Error while writing param_file: %s %s" % (e,e.message) )
            raise
        pass

    def loadParams(self,file_name):
        try:
            f = file(file_name,'rb')
            for param in self.params:
                param.set_value(cPickle.load(f))
            f.close()
            print 'params loaded from: '+file_name
        except:
            e = sys.exc_info()[0]
            print( "Error while loading param_file: %s %s" % (e,e.message) )
            raise
        pass
    
    
    def setupTraining(self,shared_data,shared_truth,learning_rate,L_reg,momentum):
        self.y = T.lvector('y')
        ind = T.lscalar()
        
#        epsilon = np.finfo(np.float32).eps
        negative_log_likelihood = -T.mean(T.log(self.py_given_x)[T.arange(self.y.shape[0]),self.y])
#        reg_term = 0
#        for param in self.Ws:
#            reg_term = reg_term+(param**2).sum()
        
        reg_term = (self.Ws[0]**2).sum()+\
                   (self.Ws[1]**2).sum()+\
                   (self.Ws[2]**2).sum()+\
                   (self.Ws[3]**2).sum()
        
        self.cost = negative_log_likelihood+L_reg*reg_term    
        
        self.error = T.mean(T.neq(self.output,self.y))
        
        updates= self.gradient_updates_momentum(self.cost,self.params,self.masks,learning_rate,momentum)
        
        self.trainModel = theano.function(inputs=[ind],
                              outputs=[self.cost,self.error],
                              updates=updates,
                              givens={self.x : shared_data[ind*self.batchsize:(ind+1)*self.batchsize,:],
                                      self.y : shared_truth[ind*self.batchsize:(ind+1)*self.batchsize]})
       
                       
        pass
    
    def setupTesting(self,shared_data,shared_truth):
#        y = T.lvector('y')
        ind = T.lscalar()
#        error = T.mean(T.neq(self.output,y))*100.
        
        self.testModel = theano.function(inputs=[ind],
                              outputs=self.error,
                              givens={self.x : shared_data[ind*self.batchsize:(ind+1)*self.batchsize,:],
                                      self.y : shared_truth[ind*self.batchsize:(ind+1)*self.batchsize]})
        
        pass
    
    def setupValidation(self,shared_data,shared_truth):
#        y = T.lvector('y')
        ind = T.lscalar()
#        errors = T.mean(T.neq(self.output,y))*100.
        
        self.validateModel = theano.function(inputs=[ind],
                              outputs=self.error,
                              givens={self.x : shared_data[ind*self.batchsize:(ind+1)*self.batchsize,:],
                                      self.y : shared_truth[ind*self.batchsize:(ind+1)*self.batchsize]})
        
        pass
   
    def gradient_updates_momentum(self,cost, params, masks, learning_rate, momentum):
    # Make sure momentum is a sane value
        assert momentum < 1 and momentum >= 0
        # List of update steps for each parameter
        updates = []
        # Just gradient descent on cost
        for param,mask in zip(params,masks):
            # For each parameter, we'll create a param_update shared variable.
            # This variable will keep track of the parameter's update step across iterations.
            # We initialize it to 0
#		 temp=np.asarray(param.get_value()*0.,dtype=theano.config.floatX)        
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            # Each parameter is updated by taking a step in the direction of the gradient.
            # However, we also "mix in" the previous step according to the given momentum value.
            # Note that when updating param_update, we are using its old value and also the new gradient step.
            updates.append((param, param - learning_rate*param_update))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            
            if mask==[]:
                mask=1
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)*mask))
        return updates
    
    
    pass


