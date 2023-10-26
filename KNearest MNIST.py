import random
import cv2
import numpy as np
import torchvision
import PIL

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

"""
function(sample_image)
sample_image = np.array(sample_image)
compare array with every number in training set and find k nearest
return what number it thinks it is
"""

def vector_distance(v1, v2):
   return np.sum((v1 - v2)**2) 

def image_to_vector(image):
    image = np.array(image, dtype=np.int32)
    image = image.flatten()
    return image




def KNearest(input, k) -> int:
    input = image_to_vector(input)
    random_samples = random.sample(range(len(mnist_train)), 1000) #random_samples currently holds 1000 numbers to use as the random sample indexs
    klist = [] #will hold list of k (vector_distance, lable) pairs
    for index in random_samples:

        image , label = mnist_train[index]
        image = image_to_vector(image)
        distance = vector_distance(input, image)

        if len(klist) < k:
            klist.append((distance, label))
        #only do this after klist is correct size k
        else:
            klist.sort()
            if klist[-1][0] > distance:
                klist[-1] = (distance, label)
    label_list = [y for x , y in klist]
    counts = np.bincount(label_list)
    return np.argmax(counts)

def test():
    test_list = random.sample(range(len(mnist_test)), 50)
    correct = 0
    for index in test_list:
        image, label = mnist_test[index]
        if KNearest(image, 1) == label:
            print("correct")
            correct += 1
        else:
            print("urdumb")
    print(correct/50)


test()