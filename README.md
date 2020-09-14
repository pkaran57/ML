# Machine Learning Models

This repository contains implementations for various Machine Learning models build using `tensorflow` and `scikit-learn`. Following is the list of contained models:
* [Perceptron](https://en.wikipedia.org/wiki/Perceptron) 
  * Path - `ml.algo.perceptron.Perceptron.py`
* [Simple Neural Network](https://en.wikipedia.org/wiki/Neural_network#:~:text=A%20neural%20network%20is%20a,of%20artificial%20neurons%20or%20nodes.) with a single hidden layer containing `N` number of hidden units
  * Path - `ml/algo/neural_net/SingleLaterNeuralNet.py`
  * [Report](https://1drv.ms/b/s!Arc54q14bwOLgtdu0xQfImRlyAABoA?e=YGW8Yu) containing results for the following experiments:
    * __Experiment #1__: Find training accuracies and plot confusion matrix for the neural network trained with a varying number of hidden units on `MNIST` dataset
    * __Experiment #2__: Train the neural network with a varying number of training samples
    * __Experiment #3__: Train the neural network with a varying number of momentum values (.25, .5 and .95)
* [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
  * Path - `ml/algo/naive_bayes/naive_bayes.py`
  * Classification accuracy of the model on the `yeast_test.txt` dataset = `44.0083%`
* [K-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) to cluster and classify the [OptDigitsdata](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) dataset
  * Path - `ml/algo/k_means/KMeans.py`
 

### Installing Dependencies
* Using `pip` - Install all the necessary requirements specified in the `requirements.txt` file by running `pip install -r requirements.txt`
* Using `pipenv` - Install all the necessary requirements specified in the `Pipfile` file by running `pipenv sync`