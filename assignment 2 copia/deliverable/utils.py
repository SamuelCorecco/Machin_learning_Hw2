import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import save_model, load_model

from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_imgs(path, folders):
    imgs = []
    labels = []
    n_imgs = 0
    for c in folders:
        # iterate over all the files in the folder
        for f in os.listdir(os.path.join(path, c)):
            if not f.endswith('.jpg'):
                continue
            # load the image (here you might want to resize the img to save memory)
            im = Image.open(os.path.join(path, c, f)).copy()
            imgs.append(im)
            labels.append(c)
        print('Loaded {} images of class {}'.format(len(imgs) - n_imgs, c))
        n_imgs = len(imgs)
    print('Loaded {} images total.'.format(n_imgs))
    return imgs, labels

def plot_sample(imgs, labels, nrows=4, ncols=4, resize=None):
    # create a grid of images
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    # take a random sample of images
    indices = np.random.choice(len(imgs), size=nrows*ncols, replace=False)
    for ax, idx in zip(axs.reshape(-1), indices):
        ax.axis('off')
        # sample an image
        ax.set_title(labels[idx])
        im = imgs[idx]
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)  
        if resize is not None:
            im = im.resize(resize)
        ax.imshow(im, cmap='gray')


# map class -> idx
label_to_idx = {
    'CHEETAH':0,
    'OCELOT': 1,
    'SNOW LEOPARD':2, 
    'CARACAL':3,
    'LIONS': 4,
    'PUMA': 5,
    'TIGER':6
}

idx_to_label = {
    0:'CHEETAH',
    1:'OCELOT',
    2:'SNOW LEOPARD', 
    3:'CARACAL',
    4:'LIONS',
    5:'PUMA',
    6:'TIGER'
}

def make_dataset(imgs, labels, label_map, img_size):
    x = []
    y = []
    n_classes = len(list(label_map.keys()))
    for im, l in zip(imgs, labels):
        # preprocess img
        x_i = im.resize(img_size)
        x_i = np.asarray(x_i)
        
        # encode label
        y_i = np.zeros(n_classes)
        y_i[label_map[l]] = 1.
        
        x.append(x_i)
        y.append(y_i)
    return np.array(x).astype('float32'), np.array(y)




def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model.h5')

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = load_model(filename)
    return model


def evaluate_model(filename_1, filename_2, x_test, y_test):
    model_nn = load_keras_model(filename_1)
    model_cnn = load_keras_model(filename_2)
    y_a = model_nn.predict(x_test)
    y_b = model_cnn.predict(x_test)
    e_a = ((y_test - y_a)**2).mean()
    e_b = ((y_test - y_b)**2).mean()
    s_a = e_a*(1-e_a)
    s_b = e_b*(1-e_b)
    T = (e_a - e_b)/(np.sqrt(s_a/y_test.shape[0] + s_b/y_test.shape[0]))
    print("ClT = ", T)
    print("variance",filename_1,"=", s_a)
    print("variance",filename_2,"=", s_b)
    if (T >= 1.96 or T <= -1.96):
        if s_a < s_b:
            print(filename_1, " model best")
            return filename_1
        else:
            print(filename_2, " model better")
            return filename_2
    print("model are same")
    return filename_2

def plot_hystory(history):
    # Print hystori of the model
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

def plot_disMat(filename, x_test, y_test, classes):
    model = load_keras_model(filename)
    y_pred = model.predict(x_test)
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    fig = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, cmap=plt.cm.Oranges)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")

def calculate_value_CM(filename, x_test, y_test, classes):
    model = load_keras_model(filename)
    y_pred = model.predict(x_test)
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    num_classes = cm.shape[0]
    tp_sum = 0
    tn_sum = 0
    fp_sum = 0
    fn_sum = 0

    for i in range(num_classes):
        tp_sum += cm[i, i]
        tn_sum += np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp_sum += np.sum(cm[:, i]) - cm[i, i]
        fn_sum += np.sum(cm[i, :]) - cm[i, i]

    print(f"TP: {tp_sum}")
    print(f"TN: {tn_sum}")
    print(f"FP: {fp_sum}")
    print(f"FN: {fn_sum}\n")