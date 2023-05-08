import numpy as np
import matplotlib.pyplot as plt

class c:
    def __init__(self, label, coordinates):
        self.label = label
        self.coordinates = coordinates


class Perceptron:
    def __init__(self, x, y, lr=0.6, max_iter=10000):
        self.x = x # input pattern -> [-bias, x, y, z]
        self.y = y # target output -> [0 / 1]
        self.lr = lr # learning rate -> 0.6
        self.w = np.random.uniform(-1, 1, np.shape(x)[1]) # initial weight
        self.max_iter = max_iter
        self.epoch = 0
        
    def predict(self, x):
        z = np.dot(x, self.w) # -w0*bias + w1x1 + x2x2 + w3x3
        return 1 if z > 0 else 0 # step
        
    def train(self):
        converge = None
        while not converge: # 1 epoch
            if self.epoch > self.max_iter:
                return # force retrun when reached max iteration
            self.epoch += 1
            converge = True
            for i in range(np.shape(self.x)[0]): # iterate through all input
                t = self.y[i] # target output
                o = self.predict(self.x[i]) # actual output
                error = t - o # error = target - output
                delta_w = np.empty(len(self.w)) # Δweight
                
                for j in range(len(self.w)): # iterate through all weight               
                    delta_w[j] = self.lr * error * self.x[i][j] 
                    self.w[j] += delta_w[j] # adjust weight and bias
                    
                    if delta_w[j] != 0:
                        converge = False # set converge flag to false if Δweight != 0                
                         
    def plot(self, c1, c2):
        w = self.w
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D([x[1] for x in c1], [x[2] for x in c1], [x[3] for x in c1], c='r')
        ax.scatter3D([x[1] for x in c2], [x[2] for x in c2], [x[3] for x in c2], c='g')
        # ax.scatter3D([-1.4],[-1.5],[2], c='c')
        
        x, y = np.linspace(-2, 2, 4), np.linspace(-2, 2, 4)
        X, Y = np.meshgrid(x, y)
        Z = (w[0] - w[1]*X - w[2]*Y) / w[3] # w1*x + w2*y + w3*z - theta = 0
        ax.plot_surface(X, Y, Z)  # Plot the plane
        plt.show()
            
def main():
    x = []
    y = []
    c1 = c(0, [[-1,0.8,0.7,1.2],[-1,-0.8,-0.7,0.2],[-1,-0.5,0.3,-0.2],[-1,-2.8,-0.1,-2]])
    for i in range(np.shape(c1.coordinates)[1]):
        x.append(c1.coordinates[i])
        y.append(c1.label)
    c2 = c(1, [[-1,1.2,-1.7,2.2],[-1,-0.8,-2,0.5],[-1,-0.5,-2.7,-1.2],[-1,2.8,-1.4,2.1]])
    for i in range(np.shape(c2.coordinates)[1]):
        x.append(c2.coordinates[i])
        y.append(c2.label)
    perceptron = Perceptron(x, y)
    perceptron.train()
    perceptron.plot(c1.coordinates, c2.coordinates)


if __name__ == "__main__":
    main()