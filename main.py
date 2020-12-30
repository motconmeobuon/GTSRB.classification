import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D



DATASET_PATH = './GTSRB'

LABELS = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', '100 km/h', '120 km/h', 'No overtaking',
          'No overtaking for tracks', 'Crossroad with secondary way', 'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track', 'Brock',
          'Other dangerous', 'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road', 'Roadwork', 'Traffic light',
          'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits', 'Only right', 'Only left', 'Only straight', 'Only straight and right', 
          'Only straight and left', 'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for track']


NUM_LABELS = 43

IMG_HEIGHT = 30
IMG_WIDTH = 30


def stat_data(data, save_path='./draft/stat_data.png'):

    print((data['ClassId'].value_counts().sort_index()))

    fig = plt.figure(figsize=(16, 4))
    plt.title('Stat')
    sns.countplot(data['ClassId'])

    plt.savefig(save_path)


def image_size_distribution(train_info, test_info, save_path='./draft/image_size_distribution.png'):

    train_dpi_subset = train_info[(train_info.Width < 80) & (train_info.Height < 80)]
    test_dpi_subset = test_info[(test_info.Width < 80) & (test_info.Height < 80)]

    g = sns.JointGrid(x='Width', y='Height', data=train_dpi_subset)

    sns.kdeplot(train_dpi_subset['Width'], train_dpi_subset['Height'], cmap='Reds',
                shade=False, shade_lowest=False, ax=g.ax_joint)
    sns.kdeplot(test_dpi_subset['Width'], test_dpi_subset['Height'], cmap='Blues',
                shade=False, shade_lowest=False, ax=g.ax_joint)

    sns.distplot(train_dpi_subset.Width, kde=True, hist=False, color='r', ax=g.ax_marg_x, label='Train distribution')
    sns.distplot(test_dpi_subset.Width, kde=True, hist=False, color='b', ax=g.ax_marg_x, label='Test distribution')
    sns.distplot(train_dpi_subset.Width, kde=True, hist=False, color='r', ax=g.ax_marg_y, vertical=True)
    sns.distplot(test_dpi_subset.Height, kde=True, hist=False, color='b', ax=g.ax_marg_y, vertical=True)

    g.fig.set_figwidth(25)
    g.fig.set_figheight(8)
    plt.savefig(save_path)


def target_class_visualization(meta_info, save_path='./draft/target_class_visualization.png'):

    sns.set_style()
    rows = 6
    cols = 8
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(25, 12))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
    
    meta_info = meta_info.sort_values(by=['ClassId'])

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx > 42:
                break
            
            img = cv2.imread(os.path.join(DATASET_PATH, meta_info['Path'].tolist()[idx]), cv2.IMREAD_UNCHANGED)
            img[np.where(img[:, :, 3]==0)] = [255, 255, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (60, 60))

            axs[i, j].imshow(img)
            axs[i, j].set_facecolor('xkcd:salmon')
            axs[i, j].set_facecolor((1.0, 0.47, 0.42))
            axs[i, j].set_title(LABELS[int(meta_info['ClassId'].tolist()[idx])])
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)

            idx += 1

    plt.savefig(save_path)


def load_data_train(data_dir):

    images = []
    labels = []

    for label in range(NUM_LABELS):
        label_dir = os.path.join(data_dir, str(label))

        for img_path in os.listdir(label_dir):
            img = load_img(os.path.join(label_dir, img_path), target_size=(30, 30))

            image = img_to_array(img)
            
            images.append(image)
            labels.append(label)

    return images, labels


def load_data_test():

    test_info_path = os.path.join(DATASET_PATH, 'Test.csv')
    test_info = pd.read_csv(test_info_path)

    images = []
    labels = []

    for index, row in test_info.iterrows():

        img = load_img(os.path.join(DATASET_PATH, row['Path']), target_size=(30, 30))

        image = img_to_array(img)
            
        images.append(image)
        labels.append(row['ClassId'])

    return images, labels


def training(save_path='./model/cnn'):

    print('Loading data...')
    images_train, labels_train = load_data_train(os.path.join(DATASET_PATH, 'Train'))

    labels_train = to_categorical(labels_train)
    print(labels_train)
    # return 0

    X_train, X_val, y_train, y_val = train_test_split(np.array(images_train), labels_train, test_size=0.2, random_state=42)

    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

    # Flattening the layer and adding Dense Layer
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(NUM_LABELS, activation='softmax'))

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    num_epochs = 30

    print('Start training...')
    start = time.time()

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, steps_per_epoch=60)

    stop = time.time()
    print(f'Training time: {stop - start}s')

    model.save(save_path)

    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, (1))
    plt.plot(history.history['accuracy'], linestyle = '-.')
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc = 'lower right')
    plt.subplot(1, 2, (2))
    plt.plot(history.history['loss'], linestyle = '-.')
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc = 'upper right')
    plt.savefig('./draft/history.png')


def testing(model_path='./model/cnn'):
    images_test, labels_test = load_data_test()
    
    X_test = np.array(images_test)
    y_test = to_categorical(labels_test)

    model = load_model(model_path)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Accuracy: ", score[1])


if __name__ == "__main__":
    # meta_info_path = os.path.join(DATASET_PATH, 'Meta.csv')
    # meta_info = pd.read_csv(meta_info_path)
    # print(meta_info.head())

    # train_info_path = os.path.join(DATASET_PATH, 'Train.csv')
    # train_info = pd.read_csv(train_info_path)
    # print(train_info.head())

    # test_info_path = os.path.join(DATASET_PATH, 'Test.csv')
    # test_info = pd.read_csv(test_info_path)
    # print(test_info.head())

    # stat_data(train_info, save_path='./draft/stat_data_train.png')
    # stat_data(test_info, save_path='./draft/stat_data_test.png')

    # image_size_distribution(train_info, test_info)

    # target_class_visualization(meta_info)

    training()
    testing()

    pass