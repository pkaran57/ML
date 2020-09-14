# Machine Learning Models

This repository contains implementation for various Machine Learning models. Following is the list of contained models:
* [Perceptron](https://en.wikipedia.org/wiki/Perceptron) 
  * Path - `ml.algo.perceptron.Perceptron.py`
  * Model outputs the following: 
    * Training and validation accuracies over epochs
    * Confusion Matrix  

* [Single Layer Neural Network](https://en.wikipedia.org/wiki/Neural_network#:~:text=A%20neural%20network%20is%20a,of%20artificial%20neurons%20or%20nodes.) with a single hidden layer containing `N` number of hidden units
  * Path - `ml/algo/neural_net/SingleLaterNeuralNet.py`
  * [Report](https://1drv.ms/b/s!Arc54q14bwOLgtdu0xQfImRlyAABoA?e=YGW8Yu) containing results for the following experiments performed on the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset:
    * __Experiment #1__: Find training accuracies and plot confusion matrix for the neural network trained with a varying number of hidden units
    * __Experiment #2__: Train the neural network with a varying number of training samples
    * __Experiment #3__: Train the neural network with a varying number of momentum values (.25, .5 and .95)
  * Model outputs the following: 
    * Training and validation accuracies over epochs
    * Confusion Matrix

* [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
  * Path - `ml/algo/naive_bayes/naive_bayes.py`
  * Classification accuracy of the model on the `yeast_test.txt` dataset = `44.0083%`

* [K-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) to cluster and classify the [OptDigitsdata](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) dataset
  * Path - `ml/algo/k_means/KMeans.py`
  * Model outputs the following: 
    * Average mean square error
    * Mean square separation
    * Mean entropy
    * Accuracy
 

### Installing Dependencies
* Using `pip` - Install all the necessary requirements specified in the `requirements.txt` file by running `pip install -r requirements.txt`
* Using `pipenv` - Install all the necessary requirements specified in the `Pipfile` file by running `pipenv sync`
