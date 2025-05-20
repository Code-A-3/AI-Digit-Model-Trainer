import os
os.environ['KERAS_HOME'] = './data'
import tensorflow as tf
import numpy as np

#################
# PREPROCESSING #
#################

# load MNIST dataset (split into training and testing sets)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print ("Training data shape: ", x_train.shape)
print ("Test data shape: ", x_test.shape)
print ("Example label: ", y_train[0])

# normalize the image data to 0-1 range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# reshape the images for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Set up augmentation
augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Generate augmented data
augmented_flow = augmenter.flow(x_train, y_train, batch_size=32)

# Create a mixed generator (50% clean + 50% augmented)
def mixed_generator(x_clean, y_clean, aug_gen):
    clean_size = len(x_clean)
    aug_iter = iter(aug_gen)
    while True:
        idx = np.random.choice(clean_size, 16, replace=False)
        clean_batch = x_clean[idx]
        clean_labels = y_clean[idx]

        aug_batch, aug_labels = next(aug_iter)

        # Combine half clean and half augmented
        batch_x = np.concatenate([clean_batch, aug_batch[:16]], axis=0)
        batch_y = np.concatenate([clean_labels, aug_labels[:16]], axis=0)

        yield batch_x, batch_y

##################
# NEURAL NETWORK #
##################

# build the convolutional neural network (CNN)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# best model saving function during the training process
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",            # filename to save
    monitor="val_accuracy",        # metric to track
    save_best_only=True,           # only save when it's the best
    verbose=1                      # prints when saving
)

# Phase 1: Train on clean data
print("Phase 1: Training on original MNIST...")
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)

# Phase 2: Train with the mixed generator
print("Phase 2: Training on augmented mixed MNIST...")
model.fit(
    mixed_generator(x_train, y_train, augmented_flow),
    steps_per_epoch=len(x_train) // 32,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)

# evaluate the model
best_model = tf.keras.models.load_model("best_model.keras")
best_loss, best_accuracy = best_model.evaluate(x_test, y_test)
print(f"\nBest saved model accuracy: {best_accuracy:.4f}")

##################
# SAVE THE MODEL #
##################

model.save("mnist_model_hybrid_20epochs.keras")
