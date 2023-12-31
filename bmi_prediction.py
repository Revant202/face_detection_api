# -*- coding: utf-8 -*-
"""Copy of bmi-prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XTyyRen4PHDtrvGfgFw4IZG_sv2zl-6w

# **BMI Prediction From Human Photograph**
---
"""

# # Commented out IPython magic to ensure Python compatibility.
# !pip install ninja
# !pip install git+git://github.com/stared/livelossplot.git

# !git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing
# # %cd /content/Self-Correction-Human-Parsing
# !mkdir checkpoints
# !mkdir inputs
# !mkdir outputs
# # %cd /content

# pip install livelossplot

# Commented out IPython magic to ensure Python compatibility.

import cv2
import time
import gdown
import numpy
import pandas
import livelossplot

from pathlib import Path
from keras import backend as K
from livelossplot import PlotLossesKeras
from sklearn.model_selection import KFold

import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing import image

import matplotlib.pyplot as plt
# %matplotlib inline

input_shape = (224, 224, 3)

"""## 1.Gathering Data

Datasets contains total of 2,272 images but consider as outlier 444 images so there is 1,828 images available for training and testing.


"""

# %cd /content/BMI-prediction-from-Human-Photograph
# col_names = ["image_filename","weight","foot","inch"]
hwcd_df = pandas.read_csv("height-weight-chart_dataset.csv")
celeb_df = pandas.read_csv("/content/celeb_datasets.csv")

df = hwcd_df.append(celeb_df, ignore_index=True)
bmi = df["weight"] * 703 / (df["foot"] * 12 + df["inch"]) ** 2

foot = df["foot"] + df["inch"] / 12
df_display = df.drop(columns=["foot", "inch"])
df_display = df_display.assign(foot=foot, bmi=bmi)
df_display.hist()
print(f"Total of datasets: {len(bmi.index)}")
print(f"Mean of BMI: {bmi.mean()}")
print(f"standard Deviation of BMI: {bmi.std()}")

df.assign(bmi=bmi)

"""## 2.Preparing the data

We are preparing the data by using *Self Correction for Human Parsing* (https://github.com/PeikeLi/Self-Correction-Human-Parsing) by only extract a human body from raw dataset which the result of this model has shown below.
<table>
  <tr>
    <td>
      Original Image:<br>
      <img src="https://drive.google.com/uc?export=view&id=11oqtX5cPf5xCU2IvYBLVFs5sqnMlfm5T" width="200">
    </td>
    <td>
      Preprocessed Image using a model:<br>
      <img src="https://drive.google.com/uc?export=view&id=1ydyIAvFcoIlAxyUago_ZkJuyfSXjEooC" width="200">
    </td>
  </tr>
</table>

And we use a preprocessed image as a filter to get rid of the background by convert all of its colors pixel to 1 and multiply them to original image.

<table>
  <tr>
    <td>
      Filter Image:<br>
      <img src="https://drive.google.com/uc?export=view&id=18rjY14s3a_i9tYWXGR-FOLu0PJ2LpkvG" width="200">
    </td>
    <td>
      Labeled Image:<br>
      <img src="https://drive.google.com/uc?export=view&id=1Bqf6gtkBdL8xhsP8hEXcMDeHMVyiu_k4" width="200" height="273">
    </td>
  </tr>
</table>

Then we resize a labeled image to 244 x 244 x 3 to suit the Resnet152 recommend input tensor.

"""

# def rgb2gray(rgb):
#     return numpy.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def preprocess_df(x, y):
  x = x.reset_index()
  X = []
  Y = []
  for i in range(x.shape[0]):

    image_filename = Path(x['image_filename'][i]).stem

    raw_filepath = 'raw_datasets/' + image_filename + ".jpg"
    preprocessed_filepath = 'raw_datasets_preprocessed/' + image_filename + ".png"
    if not Path(raw_filepath).is_file():
      raw_filepath = 'celeb_datasets/' + image_filename + ".jpg"
      preprocessed_filepath = 'celeb_datasets_preprocessed/' + image_filename + ".png"

    if Path(raw_filepath).is_file() and Path(preprocessed_filepath).is_file():
      raw_img = image.load_img(raw_filepath, target_size=(600, 400))
      raw_img = image.img_to_array(raw_img)
      preprocessed_img = image.load_img(preprocessed_filepath, target_size=(600, 400))
      preprocessed_img = image.img_to_array(preprocessed_img)

      preprocessed_img[preprocessed_img[:,:,0] > 0] = 1
      preprocessed_img[preprocessed_img[:,:,1] > 0] = 1
      preprocessed_img[preprocessed_img[:,:,2] > 0] = 1
      # img[numpy.all(img == (128, 128, 128), axis=-1)] = (0, 0, 128) # replace gray with blue
      # img[numpy.all(img == (0, 64, 0), axis=-1)] = (0, 0, 0) # replace green with black
      # img[numpy.all(img == (128, 0, 128), axis=-1)] = (0, 128, 128) # replace purple with cyan
      # img = rgb2gray(img).reshape(input_shape)

      # plt.imshow((raw_img * preprocessed_img).astype(int))
      # break

      X.append(cv2.resize(raw_img * preprocessed_img, (input_shape[0], input_shape[1])))
      Y.append(y.iloc[i])

  X = numpy.array(X)
  Y = numpy.array(Y)
  return [X, Y]

start =time.process_time()
X, Y = preprocess_df(df, bmi)
end = time.process_time()
print(f"{end - start:.2f} seconds process time")

"""### Data Normalization and Augmentation

We use ImageDataGenerator to normalize a labeled and to be augmented by rotating, shifting and horizontal flipling.
"""

train_datagen = image.ImageDataGenerator(
  samplewise_center=True,
  rotation_range=2,
  width_shift_range=0.02,
  # height_shift_range=0.02,
  # shear_range=0.02,
  # zoom_range=0.02,
  horizontal_flip=True
)
j = 0
for batch in train_datagen.flow(X, batch_size=5):
  for b in batch:
    plt.figure(j)
    plt.imshow(image.array_to_img(b))
    j += 1
  break

test_datagen = image.ImageDataGenerator(
  samplewise_center=True,
)

print(X.shape)

"""## 3.Choosing a model

* From the experiments, we decide to use ResNet152 for the best results.
* We use pre-trained weight called "imagenet" to initialize a model.
* By using it, we mean that we use it for extract a feature of human from an image so we freeze an entire model and feed an output to Fully-Connected Layer.
* A Fully-Connected Layer contains 256 neurons and using ReLu function as activation function.
* We use dropout layer with probability 0.25
* An output layer consist of only one neuron which uses linear activation function.


<img src="https://drive.google.com/uc?export=view&id=1u20brFPULhqItLZCa1zrmiHkaNKtqvMX" width="450">
"""

base_model = keras.applications.ResNet152(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape,
    pooling="avg"
)

base_model.trainable = False
base_model.summary()

"""### Extract Features

* We use k-fold cross validation to train a model by we choose k=5 and in the meanwhile, we do a features extraction before initialize our own model.
* We keep all 5 fold in the list called train_features, train_labels, test_features and test_labels for not to repeated it.
"""

batch_size=32
def extract_features(x, y, sample_count, datagen):
    features = numpy.zeros(shape=(sample_count, 2048))
    labels = numpy.zeros(shape=(sample_count))
    generator = datagen.flow(
        x, y,
        batch_size=batch_size
    )
    total = 0
    left_index = 0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        gap = features_batch.shape[0]
        total += gap
        gap_diff = total - sample_count
        gap_diff = gap_diff if gap_diff > 0 else 0
        features[left_index : left_index + (gap-gap_diff)] = features_batch[0:gap-gap_diff]
        labels[left_index : left_index + (gap-gap_diff)] = labels_batch[0:gap-gap_diff]
        left_index += gap
        if total >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features = [None]*5
train_labels = [None]*5
test_features = [None]*5
test_labels = [None]*5

k = 0
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
  X_train, Y_train, X_test, Y_test = X[train_index], Y[train_index], X[test_index], Y[test_index]

  print(f"Extract features from fold number {k+1}")
  start = time.clock()
  train_features[k], train_labels[k] = extract_features(X_train, Y_train, 2000, datagen=train_datagen)
  test_features[k], test_labels[k] = extract_features(X_test, Y_test, 2000, datagen=test_datagen)
  end = time.clock()
  print(f"Finished with {end - start:.2f} seconds process time")
  k += 1

"""## 4.Training
* Optimizer: The stochastic gradient descent (SGD) algorithm.
* Learning Rate: Start from 1e-5 and then we use the learning rate decay policy, which is implemented as follows:
\begin{equation}
LR = LR_{Base} (1 + \gamma * epoch)^{-power}
\end{equation}
where γ is 0.001 and power is 0.75. We use a momentum of 0.9
* Loss function: Huber function
* Metrics: Huber, Mean Absolute Error and R Squared
"""

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def lr_scheduler(epoch, lr):
   return 1e-5 * (1 + 0.001 * epoch) ** (-0.75)

VALIDAITON_LOSS = []

for j in range(5):
  model = Sequential()
  model.add(layers.Dense(256, activation='relu', input_dim=2048))
  model.add(layers.Dropout(0.25))
  model.add(layers.Dense(1, activation='linear'))

  model.compile(optimizer=keras.optimizers.SGD(
      learning_rate=1e-5,
      momentum=0.9
  ), loss=keras.losses.Huber(), metrics=["mean_absolute_error", coeff_determination])

  history = model.fit(train_features[j], train_labels[j], epochs=500,
    validation_data=(test_features[j], test_labels[j]),
    verbose=2,
    callbacks=[
          PlotLossesKeras(),
          keras.callbacks.LearningRateScheduler(lr_scheduler),
    ]
  )
  model.save(f"last_model{j}.h5")

  results = model.evaluate(test_features[j], test_labels[j])
  VALIDAITON_LOSS.append(results)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# axes = plt.gca()
# axes.set_xlim([0, 25])
# plt.show()

# history = model.fit(train_datagen.flow(x_train, y_train, batch_size=16),
#                     epochs=1000,
#                     validation_data=test_datagen.flow(x_test, y_test, batch_size=32),
#                     verbose=2,
#                     callbacks=[
#                           PlotLossesKeras(),
#                           keras.callbacks.LearningRateScheduler(lr_scheduler),
#                           # keras.callbacks.EarlyStopping(
#                           #     monitor='val_loss',
#                           #     patience=5,
#                           #     restore_best_weights=True
#                           # )
#                     ]
# )

"""## 5.Evaluvate

[Huber loss, MAE, R Squared] of all 5-folds
"""

VALIDAITON_LOSS

"""


We have selected the best model which MAE is equal to 4.053"""

# Commented out IPython magic to ensure Python compatibility.

# %cd /content/Self-Correction-Human-Parsing
atr_dataset_url = 'https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP'
gdown.download(atr_dataset_url, 'checkpoints/atr.pth', quiet=False)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/Self-Correction-Human-Parsing/inputs
from google.colab import files
uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/Self-Correction-Human-Parsing
!python3 simple_extractor.py --dataset 'atr' --model-restore 'checkpoints/atr.pth' --input-dir 'inputs' --output-dir 'outputs'

input_filename = Path(next(iter(uploaded))).stem

raw_input_image = cv2.imread("inputs/" +  next(iter(uploaded)))
raw_input_image = cv2.cvtColor(raw_input_image, cv2.COLOR_BGR2RGB)
raw_input_image = cv2.resize(raw_input_image, (input_shape[0], input_shape[1]))
plt.figure(0)
plt.imshow(raw_input_image)

preprocessed_input_image = image.load_img("outputs/" +  input_filename + ".png", target_size=input_shape)
preprocessed_input_image = image.img_to_array(preprocessed_input_image)
plt.figure(1)
plt.imshow(preprocessed_input_image.astype(int))

plt.figure(2)
preprocessed_input_image[preprocessed_input_image[:,:,0] > 0] = 1
preprocessed_input_image[preprocessed_input_image[:,:,1] > 0] = 1
preprocessed_input_image[preprocessed_input_image[:,:,2] > 0] = 1
plt.imshow(preprocessed_input_image)

plt.figure(3)
final_input_image = raw_input_image * preprocessed_input_image
plt.imshow(final_input_image.astype(int))

test_datagen = image.ImageDataGenerator(
  samplewise_center=True,
)

generator = test_datagen.flow(
    numpy.expand_dims(final_input_image, axis=0),
    batch_size=1
)
features_batch = base_model.predict(generator)

dependencies = {
    'coeff_determination': coeff_determination
}
model = keras.models.load_model('/content/3.935_model.h5', custom_objects=dependencies)
preds = model.predict(features_batch)
bmi_pred = preds[0][0]
print(f"BMI: {bmi_pred}")
if bmi_pred < 15:
  print("Very severely underweight")
elif 15 <= bmi_pred < 16:
  print("Severely underweight")
elif 16 <= bmi_pred < 18.5:
  print("Underweight")
elif 18.5 <= bmi_pred < 25:
  print("Normal")
elif 25 <= bmi_pred < 30:
  print("Overweight")
elif 30 <= bmi_pred < 35:
  print("Moderately obese")
elif 35 <= bmi_pred < 40:
  print("Severely obese")
elif bmi_pred >= 40:
  print("Very severely obese")