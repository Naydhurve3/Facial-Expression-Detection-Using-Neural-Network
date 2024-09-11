# Face Expression Detection Project
## Overview
This project focuses on detecting and classifying facial expressions using a Convolutional Neural Network (CNN) and Support Vector Machine (SVM) classifier. The model is trained to classify images of facial expressions, leveraging transfer learning for feature extraction.

## Key Steps
- `Dataset Preparation`: Image paths and labels are loaded from a CSV file.
- `Data Preprocessing`: Images are validated for existence, and missing images are removed. The dataset is split into training and testing subsets.
- `Model Training`: A CNN is built and trained to classify facial expressions.
- `Feature Extraction`: Features are extracted from the CNN to train an SVM for classification.
- `Model Evaluation`: Accuracy and classification metrics are calculated for both the CNN and SVM models.
- `Visualization`: Training and validation accuracy are visualized, and model predictions on test images are plotted.
## Dataset
The dataset consists of images with corresponding labels representing facial expressions (e.g., happy, sad, neutral). The image paths and labels are stored in a CSV file.

## Dataset Structure:
- `CSV File`: Contains path (relative path to the image) and label (expression label) columns.
- `Base Folder`: Contains the actual image files.
## Setup Instructions
Requirements
```
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn
- OpenCV (optional for image processing)
```
## Installation
Install the required libraries using `pip`:

Copy code
```
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```
## Code Workflow
- 1. Data Loading
The dataset is read from a CSV file and verified for missing images:

python
Copy code
```
data = pd.read_csv(csv_file_path)
data['full_path'] = data['path'].apply(lambda x: os.path.join(base_image_folder, x))
data['exists'] = data['full_path'].apply(lambda x: os.path.exists(x))
```
- 2. Data Preprocessing
The dataset is cleaned by filtering out non-existing images. Then, the data is split into training and testing sets:

python
Copy code
```
data = data[data['exists']]
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
```
- 3. Label Encoding
Labels are converted into a one-hot encoded format for the neural network:

python
Copy code
```
lb = LabelBinarizer()
train_labels_nn = lb.fit_transform(train_data['label'])
test_labels_nn = lb.transform(test_data['label'])
```
- 4. Image Data Augmentation
The images are loaded and augmented using ImageDataGenerator for training the CNN:

python
Copy code
```
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_dataframe(...)
test_gen = datagen.flow_from_dataframe(...)
```
- 5. Building the CNN Model
A CNN is built with three convolutional layers followed by dense layers:

python
Copy code
```
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    ...
    layers.Dense(len(lb.classes_), activation='softmax')
])
```
The model is compiled with the Adam optimizer and categorical cross-entropy loss:

python
Copy code
```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- 6. Training the CNN
The model is trained for 10 epochs with the training data generator:

python
Copy code
```
history = model.fit(train_gen, epochs=10, validation_data=test_gen)
```
- 7. Evaluation
The model's performance on the test data is evaluated, and the accuracy is displayed:

python
Copy code
```
test_loss, test_acc = model.evaluate(test_gen)
```
- 8. Feature Extraction
Features from the trained CNN model are extracted from the second-to-last dense layer and used to train an SVM classifier:

python
Copy code
```
feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)
train_features, train_labels_nn = extract_features_batch(train_gen, feature_extractor)
```
- 9. SVM Training
The SVM is trained on the extracted features, and its accuracy is calculated:

python
Copy code
```
svm_model = svm.SVC(kernel='linear')
svm_model.fit(train_features, train_labels_svm)
svm_accuracy = accuracy_score(test_labels_svm, svm_predictions)
```
- 10. Visualization
Accuracy plots for both training and validation are generated, and sample predictions are displayed:

python
Copy code
```
plt.plot(history.history['accuracy'], label='train accuracy')
plot_predictions(test_gen, model, lb)
```
# Results
- CNN Performance:
- Test Accuracy: ... (calculated during evaluation)
- SVM Performance:
- Test Accuracy: ...
- Classification Report:
- plaintext
  
## Conclusion
This project demonstrates a multi-step approach to facial expression classification using CNN for feature extraction and SVM for classification. Both models were trained and evaluated, providing insights into the effectiveness of using deep learning with traditional classifiers.

