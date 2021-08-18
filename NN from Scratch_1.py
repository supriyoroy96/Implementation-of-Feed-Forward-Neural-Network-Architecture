#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90

class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units):
        '''
        Initialize the neural network.
        Create weights and biases.

        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.


        Parameters
        ----------
            num_layers : Number of HIDDEN layers.
            num_units : Number of units in each Hidden layer.
        '''
        
        self.num_layers = num_layers
        self.num_units = num_units
        

        self.biases = []
        self.weights = []
        for i in range(num_layers):

            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))

            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))


            self.biases.append(np.random.uniform(-1, 1, size=(1, self.num_units)))


        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

    def __call__(self, X):
        '''
        Forward propagate the input X through the network,
        and return the output.

        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
        Returns
        ----------
            y : Output of the network, numpy array of shape m x 1
        '''
        a=X
        weights = self.weights
        biases = self.biases
        #List of input features value, all nodes value (With Activation) and output 
        self.a_state = []
        #List of all nodes value (with out Activation)
        self.h_state = []
        self.a_state.append(a)
        for i in range(len(self.weights)):
            h = np.dot(a, weights[i]) + biases[i]
            if i<len(self.weights)-1:
                self.h_state.append(h)
                a = np.maximum(h, 0) #ReLU Activation Function
                self.a_state.append(a)
            else:
                a=h
                self.a_state.append(a)
        #Output        
        y_hat=a
        return y_hat
        raise NotImplementedError

    def backward(self, X, y, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)

        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x 1
            lamda : Regularization parameter.

        Returns
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).

        Hint: You need to do a forward pass before performing bacward pass.
        '''
        

        del_W = []
        del_b = []
        weights = self.weights
        biases = self.biases
        
        y = np.reshape(y, (y.shape[0], 1))
        n = y.shape[0]
        
        
        a_state=self.a_state
        h_state=self.h_state
        
        del1 = (2/n) * (a_state[-1] - y) #derivative of (MSE Loss wrt y_hat
        
        delw = np.dot(a_state[-2].T, del1) + lamda * (weights[-1])
        
        delb = np.sum(del1, axis=0) + lamda * (biases[-1])
        
        del_W.append(delw)
        del_b.append(delb)

        
        for i in range(self.num_layers):
            del1=np.dot(del1,weights[-(i+1)].T) 
            delw=np.dot(a_state[-(i+3)].T,(del1*(h_state[-(i+1)]>0)))+lamda*(weights[-(i+2)]) #del_w for hidden layer
            delb=np.sum(del1*(h_state[-(i+1)]>0),axis=0) + lamda*(biases[-(i+2)]) #del_b for hidden Layer
            del_W.insert(0,delw)
            del_b.insert(0,delb)
        
        
        return del_W, del_b
        raise NotImplementedError


class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate):
        '''
        Create a Gradient Descent based optimizer with given
        learning rate.

        Other parameters can also be passed to create different types of
        optimizers.

        Hint: You can use the class members to track various states of the
        optimizer.
        '''
        self.learning_rate = learning_rate
        

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
        '''
        #Weight Update
        for i in range(len(weights)):
            weights[i] =weights[i] -self.learning_rate * delta_weights[i]
            biases[i] =biases[i]- self.learning_rate * delta_biases[i]

        return weights, biases
        raise NotImplementedError


def loss_mse(y, y_hat):
    '''
    Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        MSE loss between y and y_hat.
    '''
    #MSE Loss
    m = y.shape[0]
    y = np.reshape(y, (y.shape[0], 1))
    mse = np.sum(np.square(y - y_hat))
    mse=(1/m)*mse
    return mse

    raise NotImplementedError

def loss_regularization(weights, biases):
    '''
    Compute l2 regularization loss.

    Parameters
    ----------
        weights and biases of the network.

    Returns
    ----------
        l2 regularization loss 
    '''
    #Regularization Loss
    l2_reg = 0

    for i in range(len(weights)):
        l2_reg += np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))
    
    return l2_reg
    raise NotImplementedError

def loss_fn(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
        weights and biases of the network
        lamda: Regularization parameter

    Returns
    ----------
        l2 regularization loss 
    '''
    #Total Loss (MSE + Regularization)
    mse_ls=loss_mse(y, y_hat)
    reg_ls=loss_regularization(weights, biases)
    loss=mse_ls+lamda*reg_ls
    return loss
    raise NotImplementedError

def rmse(y, y_hat):
    '''
    Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        RMSE between y and y_hat.
    '''
    # RMSE Loss
    rsme = np.sqrt(loss_mse(y, y_hat))
    return rsme
    raise NotImplementedError


def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target
    ):
    '''
    In this function, you will perform following steps:
        1. Run gradient descent algorithm for `max_epochs` epochs.
        2. For each batch of the training data
            1.1 Compute gradients
            1.2 Update weights and biases using step() of optimizer.
        3. Compute RMSE on dev data after running `max_epochs` epochs.

    Here we have added the code to loop over batches and perform backward pass
    for each batch in the loop.
    For this code also, you are free to heavily modify it.
    '''

    m = train_input.shape[0]

    for e in range(max_epochs):
        epoch_loss = 0.
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            
            pred = net(batch_input)
            #print(pred)
            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss

            #print(e, i, rmse(batch_target, pred), batch_loss)
            
        #print(e, epoch_loss)
        
        #dev_pred=np.round(net(dev_input))
        #dev_rmse = rmse(dev_target,dev_pred)
        #print('Epoch',e+1,': RMSE on dev data: {:.5f}'.format(dev_rmse))
        
    dev_pred=np.round(net(dev_input))
    dev_rmse = rmse(dev_target,dev_pred)
    train_pred=np.round(net(train_input))
    train_rmse=rmse(train_target,train_pred)
    print('RMSE on train data: {:.5f}'.format(train_rmse))
    print('RMSE on dev data: {:.5f}'.format(dev_rmse))

        

def get_test_data_predictions(net, inputs):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.

    Parameters
    ----------
        net : trained neural network
        inputs : test input, numpy array of shape m x d

    Returns
    ----------
        predictions (optional): Predictions obtained from forward pass
                                on test data, numpy array of shape m x 1
    '''
    #Test Data Prediction
    pred=net(inputs)
    test_pred=np.round(pred)
    n=test_pred.shape[0]
    a,b=[],[]
    for i in range(n):
        a.append(float(i+1))
        b.append(test_pred[i][0])
    

    predictions = list(zip(a,b))

    test_data_predictions = pd.DataFrame(predictions, columns=['Id', 'Predicted'])
    
    test_data_predictions.to_csv('193109013.csv', index=False)
    
    

def read_data():
    '''
    Read the train, dev, and test datasets
    '''
    #Train Set
    df_train=pd.read_csv('dataset/train.csv')
    df_train_target=df_train[df_train.columns[0]]
    df_train_feat=df_train.drop(df_train.columns[[0]],axis=1)
    train_input=np.array(df_train_feat)
    train_target=np.array(df_train_target)

    #Dev Set
    df_dev=pd.read_csv('dataset/dev.csv')
    df_dev_target=df_dev[df_dev.columns[0]]
    df_dev_feat=df_dev.drop(df_dev.columns[[0]],axis=1)
    dev_input=np.array(df_dev_feat)
    dev_target=np.array(df_dev_target)

    #Test Set
    df_test=pd.read_csv('dataset/test.csv')
    test_input=np.array(df_test)
    

    return train_input, train_target, dev_input, dev_target, test_input


def main():

    # These parameters should be fixed for Part 1
    max_epochs = 50
    batch_size = 128


    learning_rate = 0.001
    num_layers = 1
    num_units = 64
    lamda = 0 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    get_test_data_predictions(net, test_input)


if __name__ == '__main__':
    main()


# In[ ]:




