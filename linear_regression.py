import numpy as np

class linearRegression():
    def __init__(self,dim):
        self.dim = dim
        self.weight = np.random.normal(0, 1, self.dim[1]+1)

    @staticmethod
    def predict(self, x):
        x = np.hstack((x, np.ones((x.shape[0],1))))
        y_hat = np.dot(x,weight)
        return y_hat
    
    def cost_function(self, x, y):
        y_hat = predict(x)
        cost = np.sum(np.dot(y_hat - y,y_hat - y))
        return cost
    
    def gradientDescend(self, x, y):
        x = np.hstack((x, np.ones((x.shape[0],1))))
        gradient = 2/len(x)*np.sum(np.dot(y,x)+x@x@weight)
        return gradient
    


class SGD():
    def __init__(self,model,  a,iteration, tol):
        self.model = model
        self.a = a
        self.tol = tol
        self.iteration = iteration
        self.i = 0
    
    def weight_update(self, gradient):
        if self.i < self.iteration or gradient > self.tol:
            self.model.weight -= self.a * gradient
            self.i += 1
            return True
        else:
            return False


if __name__=='__main__':
    lr = linearRegression()
    optimzier = SGD(lr, 0.01, 1000, 0.00001)
    x = np.random.randn(20,1000)
    y = [np.random.randint(0, 1) for i in range(1000)]
    epoch = 5

    from sklearn.model_selection import KFold
    KFold(n_splits=2, random_state=None, shuffle=True)

    for i in range(epoch):
        for j, (train_index, test_index) in enumerate(kf.split(x)):
            gradient = lr.gradientDescend(x[train_index], y[train_index])
            optimzier.weight_update(gradient)
        
    
        
    



    