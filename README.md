# MNIST Digit Recognition with CNN (Hybrid Training Strategy)
This project builds, trains, and evaluates a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. It uses a two-phase hybrid training strategy: training first on clean data, then on augmented data to improve generalization.

## Project Structure

### NOTE: 
This project sets: `os.environ['KERAS_HOME'] = './data'`  
This ensures that the MNIST dataset is downloaded into `./data/datasets/` instead of the global cache.
```
.
├── data
│   ├── dataset
│   │   └── mnist.npz       # training and test data set
│   └── keras.json
├── source
│   ├── create_model.py     # Full model training and saving
│   ├── run_model.py        # Loads trained model, runs predictions
│   └── general.py          # Alternate experimental training script with various visualization options
├── trained_model           # Folder to hold your saved model files
│   └── mnist_model_hybrid_20epochs.keras
├── venv                    # virtual environment (you should create one)
├── README.md
└── requirements.txt
```
## Requirements
This project was developed using:
- Python 3.11.9 (TensorFlow 2.19+ does not support Python 3.12 or later)
- TensorFlow 2.19.0
- NumPy, Matplotlib, Scikit-learn, Seaborn

## Installation
1. Copy the repo:
2. Create and run virtual environment
3. Install dependencies via requirements.txt

## How To Use
### Train Model
Run "create_model.py".
Unless you change the specifications, this will:
- Train the CNN for 10 epochs on clean MNIST data
- Continue training for 10 more epochs on a 50/50 mix of clean + augmented data
- Save the best-performing model as best_model.keras
- Also save the final model as mnist_model_hybrid_20epochs.keras

## Run the Model and Visualize Performance
Run "run_model.py"
This will:
- Load the trained model
- Predict on the test set
- Plot a confusion matrix and display overall accuracy

## Highlights
- Make sure you are using Python 3.11.9
- The MNIST data is downloaded automatically, no need to manually add files
- Two-phase hybrid training (clean → augmented)
- Data augmentation: rotation, shift, zoom
- Best model saved automatically using ModelCheckpoint
- Evaluation includes confusion matrix
- Dataset stored in ./data instead of global Keras cache
