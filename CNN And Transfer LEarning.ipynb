{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating CNN Using Scratch And Transfer Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Please download the dataset from the below url"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import the libraries as shown below\n",
    "\n",
    "from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.applications.vgg19 import VGG19\n",
    "from tensorflow.keras.applications.resnet50 import preprocess_input\n",
    "from tensorflow.keras.preprocessing import image\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img\n",
    "from tensorflow.keras.models import Sequential\n",
    "import numpy as np\n",
    "from glob import glob\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "# re-size all the images to this\n",
    "IMAGE_SIZE = [224, 224]\n",
    "\n",
    "train_path = 'cell_images/Train'\n",
    "valid_path = 'cell_images/Test'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG\n",
    "# Here we will be using imagenet weights\n",
    "\n",
    "mobilnet = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "# don't train existing weights\n",
    "for layer in mobilnet.layers:\n",
    "    layer.trainable = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "  # useful for getting number of output classes\n",
    "folders = glob('Dataset/Train/*')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Dataset/Train\\\\Parasite', 'Dataset/Train\\\\Uninfected']"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "folders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# our layers - you can add more if you want\n",
    "x = Flatten()(mobilnet.output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = Dense(len(folders), activation='softmax')(x)\n",
    "\n",
    "# create a model object\n",
    "model = Model(inputs=mobilnet.input, outputs=prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "input_1 (InputLayer)         [(None, 224, 224, 3)]     0         \n",
      "_________________________________________________________________\n",
      "block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      \n",
      "_________________________________________________________________\n",
      "block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     \n",
      "_________________________________________________________________\n",
      "block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         \n",
      "_________________________________________________________________\n",
      "block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     \n",
      "_________________________________________________________________\n",
      "block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    \n",
      "_________________________________________________________________\n",
      "block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         \n",
      "_________________________________________________________________\n",
      "block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    \n",
      "_________________________________________________________________\n",
      "block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    \n",
      "_________________________________________________________________\n",
      "block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    \n",
      "_________________________________________________________________\n",
      "block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    \n",
      "_________________________________________________________________\n",
      "block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         \n",
      "_________________________________________________________________\n",
      "block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   \n",
      "_________________________________________________________________\n",
      "block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         \n",
      "_________________________________________________________________\n",
      "block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   \n",
      "_________________________________________________________________\n",
      "block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 25088)             0         \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 2)                 50178     \n",
      "=================================================================\n",
      "Total params: 20,074,562\n",
      "Trainable params: 50,178\n",
      "Non-trainable params: 20,024,384\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# view the structure of the model\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.layers import MaxPooling2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       \n",
      "_________________________________________________________________\n",
      "max_pooling2d (MaxPooling2D) (None, 112, 112, 16)      0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 56, 56, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 50176)             0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 500)               25088500  \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 2)                 1002      \n",
      "=================================================================\n",
      "Total params: 25,100,046\n",
      "Trainable params: 25,100,046\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "### Create Model from scratch using CNN\n",
    "model=Sequential()\n",
    "model.add(Conv2D(filters=16,kernel_size=2,padding=\"same\",activation=\"relu\",input_shape=(224,224,3)))\n",
    "model.add(MaxPooling2D(pool_size=2))\n",
    "model.add(Conv2D(filters=32,kernel_size=2,padding=\"same\",activation =\"relu\"))\n",
    "model.add(MaxPooling2D(pool_size=2))\n",
    "model.add(Conv2D(filters=64,kernel_size=2,padding=\"same\",activation=\"relu\"))\n",
    "model.add(MaxPooling2D(pool_size=2))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(500,activation=\"relu\"))\n",
    "model.add(Dense(2,activation=\"softmax\"))\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "# tell the model what cost and optimization method to use\n",
    "model.compile(\n",
    "  loss='categorical_crossentropy',\n",
    "  optimizer='adam',\n",
    "  metrics=['accuracy']\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use the Image Data Generator to import the images from the dataset\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "train_datagen = ImageDataGenerator(rescale = 1./255,\n",
    "                                   shear_range = 0.2,\n",
    "                                   zoom_range = 0.2,\n",
    "                                   horizontal_flip = True)\n",
    "\n",
    "test_datagen = ImageDataGenerator(rescale = 1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 416 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "# Make sure you provide the same target size as initialied for the image size\n",
    "training_set = train_datagen.flow_from_directory('Dataset/Train',\n",
    "                                                 target_size = (224, 224),\n",
    "                                                 batch_size = 32,\n",
    "                                                 class_mode = 'categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<keras_preprocessing.image.directory_iterator.DirectoryIterator at 0x2bccf0d8488>"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 134 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "test_set = test_datagen.flow_from_directory('Dataset/Test',\n",
    "                                            target_size = (224, 224),\n",
    "                                            batch_size = 32,\n",
    "                                            class_mode = 'categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:sample_weight modes were coerced from\n",
      "  ...\n",
      "    to  \n",
      "  ['...']\n",
      "WARNING:tensorflow:sample_weight modes were coerced from\n",
      "  ...\n",
      "    to  \n",
      "  ['...']\n",
      "Train for 13 steps, validate for 5 steps\n",
      "Epoch 1/50\n",
      "13/13 [==============================] - 6s 490ms/step - loss: 1.4529 - accuracy: 0.5048 - val_loss: 0.9426 - val_accuracy: 0.4179\n",
      "Epoch 2/50\n",
      "13/13 [==============================] - 6s 498ms/step - loss: 0.7403 - accuracy: 0.6010 - val_loss: 0.6418 - val_accuracy: 0.6791\n",
      "Epoch 3/50\n",
      "13/13 [==============================] - 6s 495ms/step - loss: 0.5256 - accuracy: 0.7524 - val_loss: 0.9822 - val_accuracy: 0.5224\n",
      "Epoch 4/50\n",
      "13/13 [==============================] - 6s 459ms/step - loss: 0.4426 - accuracy: 0.7788 - val_loss: 0.4205 - val_accuracy: 0.7761\n",
      "Epoch 5/50\n",
      "13/13 [==============================] - 6s 466ms/step - loss: 0.3296 - accuracy: 0.8630 - val_loss: 0.5563 - val_accuracy: 0.7015\n",
      "Epoch 6/50\n",
      "13/13 [==============================] - 6s 473ms/step - loss: 0.3186 - accuracy: 0.8654 - val_loss: 0.3651 - val_accuracy: 0.8358\n",
      "Epoch 7/50\n",
      "13/13 [==============================] - 6s 468ms/step - loss: 0.2774 - accuracy: 0.9014 - val_loss: 0.3622 - val_accuracy: 0.7910\n",
      "Epoch 8/50\n",
      "13/13 [==============================] - 6s 473ms/step - loss: 0.2810 - accuracy: 0.8822 - val_loss: 0.3307 - val_accuracy: 0.8433\n",
      "Epoch 9/50\n",
      "13/13 [==============================] - 6s 475ms/step - loss: 0.2592 - accuracy: 0.9135 - val_loss: 0.3152 - val_accuracy: 0.8284\n",
      "Epoch 10/50\n",
      "13/13 [==============================] - 6s 463ms/step - loss: 0.2235 - accuracy: 0.9279 - val_loss: 0.2885 - val_accuracy: 0.8881\n",
      "Epoch 11/50\n",
      "13/13 [==============================] - 6s 480ms/step - loss: 0.2190 - accuracy: 0.9255 - val_loss: 0.2744 - val_accuracy: 0.8731\n",
      "Epoch 12/50\n",
      "13/13 [==============================] - 6s 464ms/step - loss: 0.2206 - accuracy: 0.9303 - val_loss: 0.3062 - val_accuracy: 0.8731\n",
      "Epoch 13/50\n",
      "13/13 [==============================] - 6s 474ms/step - loss: 0.1917 - accuracy: 0.9495 - val_loss: 0.2626 - val_accuracy: 0.8881\n",
      "Epoch 14/50\n",
      "13/13 [==============================] - 6s 484ms/step - loss: 0.2047 - accuracy: 0.9351 - val_loss: 0.3098 - val_accuracy: 0.8507\n",
      "Epoch 15/50\n",
      "13/13 [==============================] - 6s 490ms/step - loss: 0.2383 - accuracy: 0.8966 - val_loss: 0.2602 - val_accuracy: 0.9104\n",
      "Epoch 16/50\n",
      "13/13 [==============================] - 6s 481ms/step - loss: 0.2286 - accuracy: 0.9087 - val_loss: 0.3670 - val_accuracy: 0.7910\n",
      "Epoch 17/50\n",
      "13/13 [==============================] - 6s 478ms/step - loss: 0.1809 - accuracy: 0.9423 - val_loss: 0.2570 - val_accuracy: 0.8955\n",
      "Epoch 18/50\n",
      " 2/13 [===>..........................] - ETA: 5s - loss: 0.1943 - accuracy: 0.9062"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-53-69229fe26ea3>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      6\u001b[0m   \u001b[0mepochs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m50\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m   \u001b[0msteps_per_epoch\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtraining_set\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 8\u001b[1;33m   \u001b[0mvalidation_steps\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtest_set\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      9\u001b[0m )\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\util\\deprecation.py\u001b[0m in \u001b[0;36mnew_func\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    322\u001b[0m               \u001b[1;34m'in a future version'\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0mdate\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m \u001b[1;32melse\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;34m'after %s'\u001b[0m \u001b[1;33m%\u001b[0m \u001b[0mdate\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    323\u001b[0m               instructions)\n\u001b[1;32m--> 324\u001b[1;33m       \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    325\u001b[0m     return tf_decorator.make_decorator(\n\u001b[0;32m    326\u001b[0m         \u001b[0mfunc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnew_func\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'deprecated'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mfit_generator\u001b[1;34m(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)\u001b[0m\n\u001b[0;32m   1304\u001b[0m         \u001b[0muse_multiprocessing\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0muse_multiprocessing\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1305\u001b[0m         \u001b[0mshuffle\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mshuffle\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1306\u001b[1;33m         initial_epoch=initial_epoch)\n\u001b[0m\u001b[0;32m   1307\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1308\u001b[0m   @deprecation.deprecated(\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)\u001b[0m\n\u001b[0;32m    817\u001b[0m         \u001b[0mmax_queue_size\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmax_queue_size\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    818\u001b[0m         \u001b[0mworkers\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mworkers\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 819\u001b[1;33m         use_multiprocessing=use_multiprocessing)\n\u001b[0m\u001b[0;32m    820\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    821\u001b[0m   def evaluate(self,\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training_v2.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)\u001b[0m\n\u001b[0;32m    340\u001b[0m                 \u001b[0mmode\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mModeKeys\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTRAIN\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    341\u001b[0m                 \u001b[0mtraining_context\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtraining_context\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 342\u001b[1;33m                 total_epochs=epochs)\n\u001b[0m\u001b[0;32m    343\u001b[0m             \u001b[0mcbks\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmake_logs\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mepoch_logs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtraining_result\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mModeKeys\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTRAIN\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    344\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training_v2.py\u001b[0m in \u001b[0;36mrun_one_epoch\u001b[1;34m(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)\u001b[0m\n\u001b[0;32m    126\u001b[0m         step=step, mode=mode, size=current_batch_size) as batch_logs:\n\u001b[0;32m    127\u001b[0m       \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 128\u001b[1;33m         \u001b[0mbatch_outs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mexecution_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    129\u001b[0m       \u001b[1;32mexcept\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mStopIteration\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0merrors\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mOutOfRangeError\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    130\u001b[0m         \u001b[1;31m# TODO(kaftan): File bug about tf function and errors.OutOfRangeError?\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training_v2_utils.py\u001b[0m in \u001b[0;36mexecution_function\u001b[1;34m(input_fn)\u001b[0m\n\u001b[0;32m     96\u001b[0m     \u001b[1;31m# `numpy` translates Tensors to values in Eager mode.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     97\u001b[0m     return nest.map_structure(_non_none_constant_value,\n\u001b[1;32m---> 98\u001b[1;33m                               distributed_function(input_fn))\n\u001b[0m\u001b[0;32m     99\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    100\u001b[0m   \u001b[1;32mreturn\u001b[0m \u001b[0mexecution_function\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwds)\u001b[0m\n\u001b[0;32m    566\u001b[0m         \u001b[0mxla_context\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mExit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    567\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 568\u001b[1;33m       \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    569\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    570\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mtracing_count\u001b[0m \u001b[1;33m==\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_tracing_count\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36m_call\u001b[1;34m(self, *args, **kwds)\u001b[0m\n\u001b[0;32m    597\u001b[0m       \u001b[1;31m# In this case we have created variables on the first call, so we run the\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    598\u001b[0m       \u001b[1;31m# defunned version which is guaranteed to never create variables.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 599\u001b[1;33m       \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_stateless_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# pylint: disable=not-callable\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    600\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_stateful_fn\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    601\u001b[0m       \u001b[1;31m# Release the lock early so that multiple threads can perform the call\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   2361\u001b[0m     \u001b[1;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2362\u001b[0m       \u001b[0mgraph_function\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_define_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2363\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_filtered_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# pylint: disable=protected-access\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2364\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2365\u001b[0m   \u001b[1;33m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_filtered_call\u001b[1;34m(self, args, kwargs)\u001b[0m\n\u001b[0;32m   1609\u001b[0m          if isinstance(t, (ops.Tensor,\n\u001b[0;32m   1610\u001b[0m                            resource_variable_ops.BaseResourceVariable))),\n\u001b[1;32m-> 1611\u001b[1;33m         self.captured_inputs)\n\u001b[0m\u001b[0;32m   1612\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1613\u001b[0m   \u001b[1;32mdef\u001b[0m \u001b[0m_call_flat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcaptured_inputs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcancellation_manager\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[1;34m(self, args, captured_inputs, cancellation_manager)\u001b[0m\n\u001b[0;32m   1690\u001b[0m       \u001b[1;31m# No tape is watching; skip to running the function.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1691\u001b[0m       return self._build_call_outputs(self._inference_function.call(\n\u001b[1;32m-> 1692\u001b[1;33m           ctx, args, cancellation_manager=cancellation_manager))\n\u001b[0m\u001b[0;32m   1693\u001b[0m     forward_backward = self._select_forward_and_backward_functions(\n\u001b[0;32m   1694\u001b[0m         \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\function.py\u001b[0m in \u001b[0;36mcall\u001b[1;34m(self, ctx, args, cancellation_manager)\u001b[0m\n\u001b[0;32m    543\u001b[0m               \u001b[0minputs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    544\u001b[0m               \u001b[0mattrs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"executor_type\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mexecutor_type\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"config_proto\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mconfig\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 545\u001b[1;33m               ctx=ctx)\n\u001b[0m\u001b[0;32m    546\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    547\u001b[0m           outputs = execute.execute_with_cancellation(\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\tensorflow_core\\python\\eager\\execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[1;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[0;32m     59\u001b[0m     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,\n\u001b[0;32m     60\u001b[0m                                                \u001b[0mop_name\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0minputs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mattrs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 61\u001b[1;33m                                                num_outputs)\n\u001b[0m\u001b[0;32m     62\u001b[0m   \u001b[1;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     63\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mname\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# fit the model\n",
    "# Run the cell. It will take some time to execute\n",
    "r = model..fit_generator(\n",
    "  training_set,\n",
    "  validation_data=test_set,\n",
    "  epochs=50,\n",
    "  steps_per_epoch=len(training_set),\n",
    "  validation_steps=len(test_set)\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the loss\n",
    "plt.plot(r.history['loss'], label='train loss')\n",
    "plt.plot(r.history['val_loss'], label='val loss')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "plt.savefig('LossVal_loss')\n",
    "\n",
    "# plot the accuracy\n",
    "plt.plot(r.history['accuracy'], label='train acc')\n",
    "plt.plot(r.history['val_accuracy'], label='val acc')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "plt.savefig('AccVal_acc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# save it as a h5 file\n",
    "\n",
    "\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "model.save('model_vgg19.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "y_pred = model.predict(test_set)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[9.99876857e-01, 1.23175341e-04],\n",
       "       [9.99977112e-01, 2.29190919e-05],\n",
       "       [7.58346558e-01, 2.41653457e-01],\n",
       "       [9.99925494e-01, 7.45595753e-05],\n",
       "       [9.90784764e-01, 9.21520963e-03],\n",
       "       [9.12076458e-02, 9.08792377e-01],\n",
       "       [2.70295113e-01, 7.29704857e-01],\n",
       "       [3.21944878e-02, 9.67805505e-01],\n",
       "       [9.62613881e-01, 3.73861678e-02],\n",
       "       [5.13265312e-01, 4.86734688e-01],\n",
       "       [9.87143576e-01, 1.28563549e-02],\n",
       "       [9.97780263e-01, 2.21971911e-03],\n",
       "       [9.32238042e-01, 6.77619576e-02],\n",
       "       [9.21115577e-01, 7.88843632e-02],\n",
       "       [3.17853913e-02, 9.68214571e-01],\n",
       "       [1.00000000e+00, 9.47071488e-09],\n",
       "       [2.65438944e-01, 7.34561086e-01],\n",
       "       [9.99419808e-01, 5.80202264e-04],\n",
       "       [9.40651655e-01, 5.93483634e-02],\n",
       "       [9.85742450e-01, 1.42575456e-02],\n",
       "       [9.99954581e-01, 4.54339461e-05],\n",
       "       [9.99656916e-01, 3.43117106e-04],\n",
       "       [5.88945560e-02, 9.41105425e-01],\n",
       "       [9.99902725e-01, 9.72729686e-05],\n",
       "       [7.31766939e-01, 2.68233061e-01],\n",
       "       [9.99747813e-01, 2.52151629e-04],\n",
       "       [1.64933771e-01, 8.35066259e-01],\n",
       "       [9.99608934e-01, 3.91100359e-04],\n",
       "       [9.99930501e-01, 6.94626651e-05],\n",
       "       [9.82600868e-01, 1.73991676e-02],\n",
       "       [9.97824430e-01, 2.17563682e-03],\n",
       "       [8.57747495e-02, 9.14225221e-01],\n",
       "       [8.22690725e-01, 1.77309304e-01],\n",
       "       [2.20711201e-01, 7.79288828e-01],\n",
       "       [1.74620345e-01, 8.25379610e-01],\n",
       "       [9.97595489e-01, 2.40448676e-03],\n",
       "       [9.78304744e-01, 2.16952953e-02],\n",
       "       [9.96270061e-01, 3.72986798e-03],\n",
       "       [9.99968529e-01, 3.14996905e-05],\n",
       "       [2.06163585e-01, 7.93836474e-01],\n",
       "       [2.74875015e-01, 7.25125015e-01],\n",
       "       [6.29736423e-01, 3.70263547e-01],\n",
       "       [7.74357736e-01, 2.25642264e-01],\n",
       "       [9.99867320e-01, 1.32690737e-04],\n",
       "       [9.60264862e-01, 3.97351012e-02],\n",
       "       [7.71383643e-01, 2.28616387e-01],\n",
       "       [6.15290999e-01, 3.84708971e-01],\n",
       "       [9.99967098e-01, 3.28774731e-05],\n",
       "       [9.95145261e-01, 4.85474942e-03],\n",
       "       [1.08196318e-01, 8.91803682e-01],\n",
       "       [3.59025478e-01, 6.40974522e-01],\n",
       "       [1.10499725e-01, 8.89500201e-01],\n",
       "       [2.25538373e-01, 7.74461687e-01],\n",
       "       [9.98749137e-01, 1.25089986e-03],\n",
       "       [1.90371588e-01, 8.09628427e-01],\n",
       "       [9.37200725e-01, 6.27992526e-02],\n",
       "       [9.99367177e-01, 6.32850570e-04],\n",
       "       [9.99963284e-01, 3.66732384e-05],\n",
       "       [9.48564351e-01, 5.14356568e-02],\n",
       "       [9.56201553e-01, 4.37984206e-02],\n",
       "       [6.51602149e-02, 9.34839785e-01],\n",
       "       [9.99999046e-01, 1.00051784e-06],\n",
       "       [9.98620391e-01, 1.37962191e-03],\n",
       "       [9.47624370e-02, 9.05237496e-01],\n",
       "       [7.12222219e-01, 2.87777781e-01],\n",
       "       [4.08830613e-01, 5.91169357e-01],\n",
       "       [4.01257932e-01, 5.98742008e-01],\n",
       "       [9.99981642e-01, 1.84092059e-05],\n",
       "       [9.86927807e-01, 1.30722430e-02],\n",
       "       [9.73069012e-01, 2.69310307e-02],\n",
       "       [9.92525339e-01, 7.47464644e-03],\n",
       "       [4.22180533e-01, 5.77819526e-01],\n",
       "       [3.74090314e-01, 6.25909686e-01],\n",
       "       [9.00554836e-01, 9.94452089e-02],\n",
       "       [9.96229827e-01, 3.77019821e-03],\n",
       "       [8.69540453e-01, 1.30459592e-01],\n",
       "       [9.18236852e-01, 8.17631558e-02],\n",
       "       [6.71503171e-02, 9.32849705e-01],\n",
       "       [2.67355323e-01, 7.32644677e-01],\n",
       "       [9.99998689e-01, 1.25849306e-06],\n",
       "       [9.99991894e-01, 8.13550105e-06],\n",
       "       [9.63819861e-01, 3.61801200e-02],\n",
       "       [1.12400115e-01, 8.87599885e-01],\n",
       "       [8.96893084e-01, 1.03106916e-01],\n",
       "       [9.99994040e-01, 5.97917688e-06],\n",
       "       [9.99433100e-01, 5.66851115e-04],\n",
       "       [9.99959230e-01, 4.08172818e-05],\n",
       "       [9.99471962e-01, 5.28042321e-04],\n",
       "       [1.00907192e-01, 8.99092853e-01],\n",
       "       [7.78602958e-01, 2.21397042e-01],\n",
       "       [9.42606330e-01, 5.73936440e-02],\n",
       "       [9.56334770e-02, 9.04366493e-01],\n",
       "       [7.15143263e-01, 2.84856737e-01],\n",
       "       [3.28363180e-01, 6.71636820e-01],\n",
       "       [1.33568943e-01, 8.66431057e-01],\n",
       "       [7.44434819e-02, 9.25556600e-01],\n",
       "       [9.26322997e-01, 7.36770183e-02],\n",
       "       [9.69936788e-01, 3.00631654e-02],\n",
       "       [4.12148356e-01, 5.87851644e-01],\n",
       "       [9.98996079e-01, 1.00392464e-03],\n",
       "       [9.97383654e-01, 2.61637894e-03],\n",
       "       [9.99999642e-01, 3.66045072e-07],\n",
       "       [8.30568254e-01, 1.69431791e-01],\n",
       "       [1.25899151e-01, 8.74100864e-01],\n",
       "       [6.11705780e-02, 9.38829422e-01],\n",
       "       [9.99969840e-01, 3.01911859e-05],\n",
       "       [9.97462153e-01, 2.53788382e-03],\n",
       "       [8.31490874e-01, 1.68509126e-01],\n",
       "       [2.56192029e-01, 7.43807971e-01],\n",
       "       [9.99974728e-01, 2.52405061e-05],\n",
       "       [7.60781288e-01, 2.39218742e-01],\n",
       "       [5.35193384e-01, 4.64806587e-01],\n",
       "       [9.99709666e-01, 2.90352647e-04],\n",
       "       [2.36094102e-01, 7.63905883e-01],\n",
       "       [9.99740064e-01, 2.59921653e-04],\n",
       "       [9.99999285e-01, 6.80445339e-07],\n",
       "       [1.00000000e+00, 8.55684501e-09],\n",
       "       [9.99842644e-01, 1.57280505e-04],\n",
       "       [9.69608068e-01, 3.03919483e-02],\n",
       "       [9.99734938e-01, 2.65098090e-04],\n",
       "       [8.61145318e-01, 1.38854623e-01],\n",
       "       [8.84838045e-01, 1.15161963e-01],\n",
       "       [1.26957521e-01, 8.73042524e-01],\n",
       "       [1.08776897e-01, 8.91223073e-01],\n",
       "       [9.85367537e-01, 1.46324970e-02],\n",
       "       [9.99420404e-01, 5.79623622e-04],\n",
       "       [2.22981453e-01, 7.77018547e-01],\n",
       "       [1.97749451e-01, 8.02250564e-01],\n",
       "       [9.99945760e-01, 5.42549496e-05],\n",
       "       [2.38410935e-01, 7.61589050e-01],\n",
       "       [9.18569207e-01, 8.14308450e-02],\n",
       "       [9.98411298e-01, 1.58868567e-03],\n",
       "       [1.78226024e-01, 8.21774006e-01],\n",
       "       [1.95776328e-01, 8.04223716e-01]], dtype=float32)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "y_pred = np.argmax(y_pred, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,\n",
       "       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,\n",
       "       1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,\n",
       "       1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,\n",
       "       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,\n",
       "       1, 1], dtype=int64)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing import image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=load_model('model_vgg19.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "img=image.load_img('Dataset/Test/Uninfected/2.png',target_size=(224,224))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.]],\n",
       "\n",
       "       [[0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.]],\n",
       "\n",
       "       [[0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.]],\n",
       "\n",
       "       [[0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.]],\n",
       "\n",
       "       [[0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        ...,\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.],\n",
       "        [0., 0., 0.]]], dtype=float32)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=image.img_to_array(img)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(224, 224, 3)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=x/255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1, 224, 224, 3)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=np.expand_dims(x,axis=0)\n",
    "img_data=preprocess_input(x)\n",
    "img_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.01155142, 0.98844856]], dtype=float32)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(img_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "a=np.argmax(model.predict(img_data), axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Uninfected\n"
     ]
    }
   ],
   "source": [
    "if(a==1):\n",
    "    print(\"Uninfected\")\n",
    "else:\n",
    "    print(\"Infected\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
