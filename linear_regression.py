import numpy as np
import pandas as pd

class linearRegression():
    def __init__(self,dim):
        self.dim = dim
        self.weight = np.random.normal(0, 1, self.dim[1]+1)

    def predict(self, x):
        x = np.hstack((x, np.ones((x.shape[0],1))))
        y_hat = np.dot(x,self.weight)
        return y_hat
    
    def check_g(self,x):
        x = np.hstack((x, np.ones((x.shape[0],1))))
        y_hat = np.dot(x,self.weight)
        cost = np.sum(np.dot(y_hat - y,y_hat - y))/x.shape[0]

        gradient = []
        for i in range(len(self.weight)-1):
            weight_update = np.add(self.weight, [-0.01 if j==i else 0 for j in range(x.shape[1])])
            y_hat_check = np.dot(x,weight_update)
            cost_check = np.sum(np.dot(y_hat_check - y,y_hat_check - y))/x.shape[0]
            gradient.append(((cost-cost_check)/0.01))
        print('gradient', (cost-cost_check)/0.01)



    
    def cost_function(self, x, y):
        y_hat = self.predict(x)
        cost = np.sum(np.dot(y_hat - y,y_hat - y))/x.shape[0]
        return cost
    
    def gradientDescend(self, x, y):
        x = np.hstack((x, np.ones((x.shape[0],1))))
        gradient = 2 / x.shape[0] * (np.dot(y.T,x) + x.T @ x @ self.weight)
        return gradient
    


class SGD():
    def __init__(self,model,a, tol):
        self.model = model
        self.a = a
        self.tol = tol
    
    def weight_update(self, gradient):
        if np.linalg.norm(gradient) > self.tol:
            self.model.weight -= self.a * gradient
            return True
        else:
            return False


if __name__=='__main__':
    np.random.seed(1)
    x1 = np.vstack((np.random.normal(4,2,500),np.random.normal(4, 0.5, 500))).reshape(500,2)
    y1 = np.array([0] * 500).reshape(500,1)
    x1_update = np.hstack((x1,y1))
    x2 = np.vstack((np.random.normal(0,2,500),np.random.normal(0, 0.5, 500))).reshape(500,2)
    y2 = np.array([1] * 500).reshape(500,1)
    x2_update = np.hstack((x2,y2))
    dataset = np.concatenate((x1_update,x2_update))
    x = dataset[:,:-1]
    y = dataset[:,-1]

    epoch = 100
    lr = linearRegression(x.shape)
    optimzier = SGD(lr, 0.0001, 0.0000000001)

    stop = True
    for i in range(epoch):
        arr = np.arange(1000)
        np.random.shuffle(arr)
        arr = arr.reshape((10,100))

        for choice in arr:
            x_train = dataset[choice, :-1]
            y_train = dataset[choice, -1]
            gradient = lr.gradientDescend(x_train, y_train)
            stop = optimzier.weight_update(gradient)
            if not stop:
                break
        if not stop:
            break    
        print(lr.cost_function(x,y))
    
        
    



    