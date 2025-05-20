import os
os.environ['KERAS_HOME'] = './data'
import tensorflow as tf
import matplotlib.pyplot as plt

#################
# PREPROCESSING #
#################

# load MNIST dataset (split into training and testing sets)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print ("Training data shape: ", x_train.shape)
print ("Test data shape: ", x_test.shape)
print ("Example label: ", y_train[0])

# #show the first image
# plt.imshow(x_train[0], cmap='gray')
# plt.title(f"Label: {y_train[0]}")
# plt.axis('off')
# plt.show()

# normalize the image data to 0-1 range
x_train = x_train / 255.0
x_test = x_test / 255.0

# # visualize the first 12 digits in the training set
# plt.figure(figsize=(10,4))
# for i in range(12):
#     plt.subplot(2,6,i+1)
#     plt.imshow(x_train[i], cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

#flatten the images
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

##################
# NEURAL NETWORK #
##################

# # build the dense model
# model = keras.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(784,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# alternatively build the convolutional neural network (CNN)
model = tf.keras.Sequential([
    tf.layers.Reshape((28,28,1), input_shape=(784,)),
    tf.layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    tf.layers.MaxPooling2D(pool_size=(2,2)),
    tf.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    tf.layers.MaxPooling2D(pool_size=(2,2)),
    tf.layers.Flatten(),
    tf.layers.Dense(64, activation='relu'),
    tf.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

##################
# save the model #
##################

model.save("mnist_model.keras")

##################################
# make predictions and visualize #
##################################

# import numpy as np

# # predict the first 12 test images
# predictions = model.predict(x_test[:12])

# # show results
# plt.figure(figsize=(10,4))
# for i in range(12):
#     plt.subplot(2,6, i + 1)
#     # reshape back to 28x28 for display
#     image = x_test[i].reshape(28,28)
#     predict_label = np.argmax(predictions[i])
#     true_label = y_test[i]

#     plt.imshow(image, cmap='gray')
#     plt.title(f"P: {predict_label}\nT: {true_label}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

# # get incorrect predictions (debugging)
# predictions = model.predict(x_test)
# predicted_classes = np.argmax(predictions, axis=1)
# incorrect = np.where(predicted_classes != y_test)[0]

# # show load of errors per digit
# error_list = {i: 0 for i in range(10)}

# for index in incorrect:
#     error_list[y_test[index]] += 1

# print (error_list)

# # show first 12 wrong predictions
# plt.figure(figsize=(10,4))
# for index, i in enumerate(incorrect[:12]):
#     plt.subplot(2,6, index + 1)
#     image = x_test[i].reshape(28,28)
#     plt.imshow(image, cmap='gray')
#     plt.title(f"True: {y_test[i]}, Pred: {predicted_classes[i]}")
#     plt.axis('off')

# plt.tight_layout()    
# plt.show()

#########################
# plot confusion matrix #
#########################

# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # create confusion matrix
# conf_matrix = confusion_matrix(y_test, predicted_classes)

# # plot
# plt.figure(figsize=(8,6))
# sns.heatmap(conf_matrix,annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()