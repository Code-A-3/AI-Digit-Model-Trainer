import os
os.environ['KERAS_HOME'] = './data'
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# run model
model = keras.models.load_model("./trained_model/mnist_model_epoch10_augmented.keras")
predictions = model.predict(x_test)

# plot confusion matrix & show accuracy
predicted_classes = np.argmax(predictions, axis=1)
conf_matrix = confusion_matrix(y_test, predicted_classes)

accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f"Confusion Matrix | Accuracy: {accuracy:.4f}")
plt.show()