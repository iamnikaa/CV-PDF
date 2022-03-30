from array import array
import pickle
from itertools import combinations
import random
import os
import sys
import logging
import json
import pickle

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# # from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from tensorflow.keras.models import Model

from keras.models import Sequential, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense
#from keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.optimizers import Adam
import keras
import tensorflow
from tensorflow.keras.applications import ResNet101,ResNet50V2
#	import tensorflow.keras.utils.Sequence

log = logging.getLogger()
log.setLevel(logging.DEBUG)




random.seed(42)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)


checkpointer = tensorflow.keras.callbacks.ModelCheckpoint(filepath="Model_ep_new_weights.{epoch:02d}-{val_loss:.6f}.hdf5", monitor='val_loss', verbose=1, save_best_only=False)
earlystopper = tensorflow.keras.callbacks.EarlyStopping(patience=39, verbose=1)
reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,patience=8, min_lr=0.000001, verbose=1)


known_doc_list = [
    [1, 2, 3],
    [4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35],
    [36, 37, 38],
    [39, 40],
    [41, 42],
    [43, 44],
    [45, 46, 47],
    [48],
    [49],
    [50, 51],
    [52],
]

img_folder = "/home/konverge/Desktop/work/25_Mar_DOC/processed_new/"
resized_img_folder = "/home/konverge/Desktop/work/Doc/proj/resized_imgs/"


class DataGenerator(tensorflow.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, samples_ids, labels, features, batch_size=32, dim=(200704), n_channels=1,
                 n_classes=2, shuffle=True, gen_type = 'train'):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.samples_ids = samples_ids
        self.labels = labels
        self.features = features
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.gen_type = gen_type
        self.on_epoch_end()
        log.info("label shape is %s", self.samples_ids.shape)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.samples_ids) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        print("guess what")
        self.indexes = np.arange(len(self.samples_ids))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, index, tell_indices=False):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_samples = [self.samples_ids[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        if tell_indices:
            log.info("indexes are %s", indexes)
            log.info("samples are %s", batch_samples)
            log.info("labels are %s", batch_labels)

        # Generate data
        X, y = self.__data_generation(batch_samples, batch_labels, tell_indices)

        return X, y


    def __data_generation(self, batch_samples, batch_labels, tell_indices=False):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim)
        # Initialization

        X = []
        y = []

        # Generate data
        for i, ID in enumerate(zip(batch_samples, batch_labels)):
            ID_x, ID_y = ID
            # Store sample
            if tell_indices:log.info("ID %s label %s", ID_x, ID_y)
            np.random.shuffle(ID_x)

            X.append(self.features[ID_x].flatten())

            # Store class
            y.append(self.labels[ID_y])
        
        y_ = tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
        if tell_indices: log.info("%s labeled %s",y, y_)

        return np.array(X), y_


def generate_groups(doc_index):
    pages = sorted(list(doc_index.keys()))
    # log.info("pages range %s %s", min(pages), max(pages))
    same_group = []
    diff_group = []
    for c in combinations(pages, 2):
        if doc_index[c[0]] == doc_index[c[1]]:
            # 0 if same document, 1 if different
            same_group.append([*c, 0])
        else:
            diff_group.append([*c, 1])
    same_group = same_group*38
    log.info("same %s", len(same_group))
    log.info("diff %s", len(diff_group))

    # Oversample same_group

    

    return same_group+diff_group


def setup():
    with open(r"/home/konverge/Desktop/work/25_Mar_DOC/new_project2/Doc_file.txt") as f:
        t = f.read()
    print("Setup started")
    doc_pages = []
    for line_num, line in enumerate(t.split("\n")):
        doc_pages.append((line_num, list(
            map(int, filter(lambda x: True if len(x) else False, line.strip().split(","))))))

    doc_index = {}
    doc_pages = doc_pages[:-1]
    # reset to zero-indexed document page numbers
    doc_pages = [(x, list(map(lambda a:a, y))) for x, y in doc_pages]
    for elem in [(x[1], x[0]) for x in doc_pages]:
        for j in elem[0]:
            doc_index[j] = elem[1]

    with open(r"/home/konverge/Desktop/work/25_Mar_DOC/new_project2/doc_pages2100epw.pkl", "wb") as f:
        pickle.dump(doc_pages, f)

    dataset = generate_groups(doc_index)
    # IMPORTANT: shuffle
    random.shuffle(dataset)
    dataset = np.array(dataset)

    return dataset


def resize_and_cut(img):
    n = 1
    img = cv2.resize(img,(6600, 5100))
    im = img.astype("float32")
    avg_im = (im[0:im.shape[0]//3, :, :] + im[im.shape[0]//3:2*im.shape[0]//3, :, :] + im[2*im.shape[0]//3:im.shape[0], :, :])//3
    folded_im = (avg_im[:, 0:avg_im.shape[1]//3, :] + avg_im[:, avg_im.shape[1]//3:2*avg_im.shape[1]//3, :] + avg_im[:, 2*avg_im.shape[1]//3:avg_im.shape[1], :])//3
    shape = folded_im.shape
    # print(shape)
    crop_dims_x,crop_dims_y = shape[0]//2,  shape[1]//2
    # Take center 448 pixels and downsize - document is less likely to be empty there
    # img = folded_im[crop_dims_x-224:crop_dims_x+224, crop_dims_y-224:crop_dims_y+224, :]
    img = cv2.resize(folded_im, (224*n, 224*n))
    return img


def infer():
    print("Infer Started")
    # images = [(img_folder+x, resized_img_folder+x)
    #          for x in os.listdir(img_folder) ] #if x.endswith(".png")
    resnet = ResNet50V2(weights="imagenet",
                              include_top=False)



    x = Dense(1024, activation = 'relu')(resnet.output)
    x = tensorflow.keras.layers.Dropout(.4)(x)
    x = Dense(512, activation = 'softmax')(x)
    x = tensorflow.keras.layers.Dropout(.2)(x)
    x = Dense(512, activation = 'relu')(x)
    prediction = Dense(2048, activation = 'softmax')(x)
    model = Model(inputs = resnet.input, outputs = prediction)
    #print(".......................",images)
    
    # image_list = []
    
    # for i, o in images:
    #     img = resize_and_cut(cv2.imread(i))
    #     #img = cv2.resize(img, (224,224))
    #     image_list.append(img)
    # try:
    #     with open(r"/home/konverge/Desktop/work/25_Mar_DOC/dataImagefullw.pkl", "wb") as f:
    #         pickle.dump(image_list, f)
    # except:
    #     pass
    with open(r"/home/konverge/Desktop/work/25_Mar_DOC/dataImagefullw.pkl", "rb") as input_file:
        image_list = pickle.load(input_file)

    final_features = []
    print("length:",len(image_list))
    print("before batching")
    for i in range(0,len(image_list),64):
        arr = []
        for batch in range(64):
            try:
                arr.append(image_list[batch+(i)])
            except:
                continue
        image_array = np.array(arr)
        features = model.predict(image_array)

        for j in features:
            final_features.append(j)

    
    #print(image_array)
    #log.info("image shape %s", image_array.shape)
    #features = model.predict(image_array)
    final_features = np.array(final_features)
    with open(r"/home/konverge/Desktop/work/25_Mar_DOC/new_project2/s4o1final_featuresw.pkl", "wb") as f:
        pickle.dump(final_features, f)
    
    features = final_features.reshape((final_features.shape[0], 7 * 7 * 2048))
    #log.info("features shape %s", features.shape)
    return image_array, features


def secondary_model():
    model = Sequential([
        Dense(256, input_shape=(2 * 7 * 7 * 2048,), activation="tanh", kernel_initializer = 'glorot_uniform'),
        Dense(16, activation="tanh", kernel_initializer = 'glorot_uniform'),
        Dense(2, activation="softmax", kernel_initializer = 'glorot_uniform')
    ])
    return model


def get_one_hot(category, size=2):
    assert category < size
    return np.array([1 if cn == category else 0 for cn, x in enumerate([0]*size)])


def data_generator(X, y, features):

    # log.info("x shape %s", X.shape)
    # log.info("y shape %s", y.shape)

    assert len(X) == len(y)
    # while True:
    idx = random.choice(range(len(X)-1))
    X_l, y_l = [], []
    for batch_size in range(2):
        X_l.append(features[X[idx], :].flatten())
        y_l.append(get_one_hot(y[idx]))
    X_, y_ = np.array(X_l), np.array(y_l)
    # log.info("batch x shape %s, batch y shape %s", X_.shape, y_.shape)
    yield X_, y_


def train():
    dataset = setup()
    new_x = []
	#count = 0
    #print(type(dataset))
    X, y = dataset[..., :2], dataset[:, 2]
    
	
	
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)
    image_data, features = infer()
    second_model = secondary_model()
    second_model.compile(optimizer=Adam(learning_rate=3e-6),
                        loss='binary_crossentropy')
    print(set(y_test))
    #val_data = [(X, y) for X, y in DataGenerator(samples_ids=X_test, labels=y_test,features=features)]
    val_generator = DataGenerator(samples_ids=X_test, labels=y_test,features=features, gen_type = 'val')
    #H = second_model.fit_generator(data_generator(X_train, y_train, features), epochs = 20, steps_per_epoch=10, validation_data=val_data)

    H = second_model.fit_generator(DataGenerator(samples_ids=X_train, labels=y_train,
                                                features=features, gen_type='train'), epochs=80, steps_per_epoch=len(X_train)/64, validation_data=val_generator,callbacks=[checkpointer,earlystopper,reduce_lr])
    
	#H = second_model.fit_generator(DataGenerator(samples_ids=X_train, labels=y_train,features=features, gen_type='train'), epochs=200, steps_per_epoch=30, validation_data=val_generator)

    j = second_model.to_json()
    with open("/home/konverge/Desktop/work/25_Mar_DOC/new_project2/secondary_model3.json","w") as f:
        f.write(j)
        
    second_model.save("/home/konverge/Desktop/work/25_Mar_DOC/new_project2/diff_predictor_40ep.h5")
    return second_model


def test(weights_path = '', test_all=True):
    print("bgin")
    if len(weights_path):
        second_model = load_model(weights_path)
    else:
        second_model = secondary_model()
    second_model.compile(optimizer=Adam(learning_rate=3e-4),
                        loss='binary_crossentropy', metrics = ['accuracy'])
    dataset = setup()
    X, y = dataset[..., :2], dataset[:, 2]
    _, features = infer()
    test_generator = DataGenerator(samples_ids=X, labels=y,features=features, gen_type = 'test', shuffle=False)
    if test_all:
        score = second_model.evaluate_generator(test_generator, len(y)//32, workers=1)
        print(score)
        print(second_model.metrics_names)
        print("%s: %.2f%%" % (second_model.metrics_names[1], score[1]*100))
    # print(dir(test_generator))
    # y_true = test_generator.classes
    else:
        item = test_generator.__getitem__(50, tell_indices=True)
        preds = np.argmax(second_model.predict(item[0]), axis=1)
        preds = [get_one_hot(x) for x in preds]
        trues = item[1]
        print(confusion_matrix(np.argmax(np.array(trues),axis=1),np.argmax(np.array(preds), axis=1)))

        for i,j in zip(preds, trues):
            print("pred", i, "true", j)
        
train()

#print("\n\n\n\n\n\n\n\n\n\n\n\n\n",x)
#test(weights_path="/home/konverge/Desktop/Work/foundationai/diff_predictor_decent_model_4_12.h5")

# [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
# [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]

