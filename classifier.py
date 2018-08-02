from matplotlib import pyplot as plt
import numpy as np


def svm_learn(X,Y):
    # Initialize the weights
    w = np.zeros(len(X[0]))
    # Learning rate for the support vector machine
    eta = 1
    # Epochs (number of learning iterations)
    epochs = 10000
    # Errors
    errors = []

    # begin the training
    for epoch in range(1, epochs):
        error = 0
        for i,x in enumerate(X):
            # hinge loss function
            # loss = 1 - (y*f(x)) { if y*f(x) >= 1, then loss = 0 } This is just to make sure that know when we have the correct classification
            # Think about it for a second:
            #   y = 1, for some x, then if f(x) = 0.8 then loss = 0.2, so we are not quite there.
            #   y = -1, for some x, then if f(x) = -1, then loss = 0
            #   y = 1, for some x, then if f(x) = 2, then loss = 0
            # We are going to attempt to minimize the gradients for the hinge loss function
            # The hinge loss function basically just tells us wether we have classified something correctly or not, and by how much
            
            # This happens if we have misclassified given our current weights
            if(Y[i]*np.dot(X[i], w) < 1):
                # We know that we have misclassified, because if we had correctly classified, 
                # then Y[i] would have shot us in the positive direction above on (-1 * -1 would be a positive, and 1*1 would be a positive, both being greater than 1)
                # We then know that we can update our approximation function considering the direction of Y, and the vector X, and the current gradient of w
                w = w + eta*((X[i]*Y[i] - (2/epoch)*w));
                error = 1
            else: # in the case that we have created something bigger than one, 
                  # we actually dont need it to be bigger than one, 
                  # so we know that the direction for which we are moving the gradients is to be less than what was before
                # we know that we only need to attempt to increase the margin for our hyper plane
                w = w - (2/epoch)*eta*w;
            
            errors.append(error);

            # NB! 
            #   Notice how the more epochs, the less the affect of the weight update
            #   Notice, we are essentially updating for the direction of the vector by using Y*X, giving smaller and smaller updates every time
                
    # Show some interesting information concerning the misclasifications and the epochs
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w;
            


if __name__ == '__main__':
    #Input data - [X value, Y value, Bias term]
    X = np.array([
        [-2,4,-1],
        [4,1,-1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],
    ]);

    #Associated output labels - First 2 examples are labeled '-1' and last 3 are labeled '+1'
    y = np.array([-1,-1,1,1,1])

    for d, sample in enumerate(X):
        if(d<2):
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2);
        else:
            plt.scatter(sample[0], sample[1], s=120, marker="+", lineWidths=2);

    

    plt.plot([-2,6],[6,0.5]);
    plt.show();

    weights = svm_learn(X, y);
