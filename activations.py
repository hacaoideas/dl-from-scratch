import numpy as np 


class Sigmoid():
    def __call__(self, x):
        return 1/(1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x)*(1-self.__call__(x))

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x-np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True))
        #axis 0 is the first axis
        #axis -1 is the last one
        
    
    def gradient(self, x):
        p = self.__call__(x)
        return p*(1-p)



if __name__ == "__main__":
    x = np.linspace(0, 2, 5)
    print(x)
    Activation = Sigmoid()
    print(Activation(x))
    print(Activation.gradient(x))