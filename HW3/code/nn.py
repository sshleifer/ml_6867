import numpy as np

def softmax(x):
    return np.max(x, 0)

class NN(object):
    
    def __init__(self, n_hidden_layers=2, n_hidden_nodes=2, activation_func=softmax, epochs=10):
        self.z = np.zeros((n_hidden_layers + 2, n_hidden_nodes))
        self.a = np.zeros((n_hidden_layers + 2, n_hidden_nodes))
        self.w = np.ones((n_hidden_layers + 2, n_hidden_nodes))
        self.n_layers =  n_hidden_layers + 2
        self.activate = activation_func
        self.epochs = epochs
        self.bias = np.zeros(n_hidden_layers)
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.d = np.array((n_hidden_layers, n_hidden_nodes, n_hidden_layers)) # matrix of errors for each layer


    def predict(self, x=None):
        '''pass one x row through the network'''
        x=X[2]
        self.a[0] = x
        for layer in range(1, self.n_hidden_layers+1):
            #print layer#, self.a, self.z
            #print self.a[layer]
            self.z[layer] = self.w[layer].T.dot(self.a[layer - 1]) # aggregate
            self.a[layer] = self.activate(self.z[layer])
                
        # final layer
        return self.activate(self.a[layer])
    
    #def calc_error(self, x, weight):
        
    
    def fit(self, X, y):
        for epoch in range(self.epochs):
            i = np.random.randint(0, len(X))
            x, target = X[i], y[i]
            prediction =  self.predict(x)
            self.backprop(target - error)
            
    def backprop(self, error):
        
        return
        # pseudocode
        
    def error_mat(self):
        #np.diag()
        return