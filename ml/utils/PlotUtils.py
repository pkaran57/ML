import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

img_counter = 0

def plot_accuracy(training_accuracy_over_epochs, validation_accuracy_over_epochs, plot_title_attributes):
    """
    Plot accuracies over epochs for training and validation samples
    :param training_accuracy_over_epochs:
    :param validation_accuracy_over_epochs:
    :param plot_title_attributes:
    :return:
    """
    title = "Accuracy over Epochs\n"
    for key, value in plot_title_attributes.items():
        title += '{}={}, '.format(key, value)

    plt.title(title)

    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy (%)')

    plt.plot(training_accuracy_over_epochs, label='Training Accuracy')
    plt.plot(validation_accuracy_over_epochs, label='Validation Accuracy')

    global img_counter
    plt.savefig('out/{}-accuracy.svg'.format(img_counter), format='svg', dpi=1200)
    img_counter += 1

    plt.legend()
    plt.show()


def plot_confusion_matrix(validation_samples, get_prediction_label_function, plot_title_attributes, display_labels=range(10)):
    """
    Plot confusion matrix
    :param validation_samples:
    :param get_prediction_label_function:
    :param plot_title_attributes:
    :param display_labels:
    :return:
    """
    predicted_labels = []
    actual_labels = []

    for validation_sample in validation_samples:
        predicted_target_label = get_prediction_label_function(validation_sample)
        predicted_labels.append(predicted_target_label)
        actual_labels.append(validation_sample.true_class_label)

    training_samples_confusion_matrix = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)

    confusion_matrix_display = ConfusionMatrixDisplay(training_samples_confusion_matrix, display_labels=display_labels)
    confusion_matrix_display.plot(values_format='d')

    title = "Confusion matrix\n"

    for key, value in plot_title_attributes.items():
        title += '{}={}, '.format(key, value)

    global img_counter
    plt.savefig('out/{}-confusion-matrix.svg'.format(img_counter), format='svg', dpi=1200)
    img_counter += 1

    plt.title(title)
    plt.show()
