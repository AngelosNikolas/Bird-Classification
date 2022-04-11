# %% [markdown]
# # 250 species of birds classification Using the VGG16 Pre-trained Model
#
# ## Angelos Nikolas
#
# ### The dataset is available at: https://www.kaggle.com/datasets/gpiosenka/100-bird-species/code

# %%
# Importing libraries
import cv2
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
import random
import tensorflow as tf

# %%
# Setting seed
random.seed(42)

# %% [markdown]
# # Processing Data

# %%
# Setting up the data and splitting it into training and test sets
generator = tf.keras.preprocessing.image.ImageDataGenerator()

train = generator.flow_from_directory(
    "/home/azureuser/localfiles/birdsdata/train/", class_mode='categorical', batch_size=32, target_size=(130, 130))
test = generator.flow_from_directory("/home/azureuser/localfiles/birdsdata/test/",
                                     class_mode='categorical', batch_size=32, target_size=(130, 130))
valid = generator.flow_from_directory(
    "/home/azureuser/localfiles/birdsdata/valid/", class_mode='categorical', batch_size=32, target_size=(130, 130))

# %% [markdown]
# # Build transfer learning model

# %%

base_model = VGG16(
    weights='imagenet',
    input_shape=(130, 130, 3),
    include_top=False)

# freezing the model
base_model.trainable = False
# Specifying the input shape
inputs = tf.keras.Input(shape=(130, 130, 3))

x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalMaxPooling2D()(x)
# Output layer is set to 250 for the classification of the 250 bird classes
outputs = tf.keras.layers.Dense(250, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

model.summary()

# %% [markdown]
# # Fitting the model

# %%
history = model.fit(train, validation_data=valid, epochs=5)

# %% [markdown]
# # Fine-Tunning

# %%
# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=tf.keras.optimizers.Adam(0.00001),  # Very low learning rate
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train end-to-end. Be careful to stop before you overfit!
model.fit(train, epochs=5, validation_data=valid)

# %%
model.save("/home/azureuser/cloudfiles/code/Users/c1044433/Task2/model.h5")

# %% [markdown]
# # Results

# %%

y_true = test.labels
y_pred = model.predict(test).argmax(axis=1)

# %%
acc_test = model.evaluate(test)[1]
print("The testing accuracy is: ", acc_test)

# %%
%pip install seaborn

conf = confusion_matrix(y_true=y_true, y_pred=y_pred)

conf_plt = sns.heatmap(conf, fmt='', cmap='Blues_r')

# %% [markdown]
# # Bird Classifier
#
# ### Import a bird and return the species it belongs to

# %%
# Create the dictionary
classes = test.class_indices
lab = {i: u for u, i in classes.items()}
print(lab)

# %%

sys.path.append('/usr/local/lib/python2.7/site-packages')

# %%

# Provide the image
img = '/home/azureuser/cloudfiles/code/Users/c1044433/Task2/TestImages/bird.jpg'
pic = cv2.imread(img)


imga = cv2.resize(pic, (130, 130))

res = VGG16_model.predict(np.array([imga]))
res = np.argmax(res)

plt.imshow(pic)
print(lab[res])
# output(img)
