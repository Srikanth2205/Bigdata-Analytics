import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import os
# this is done to avoid error due to eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def process_images_labels(image_path, img_folders, mapper):
    """
    Process the image such that resnet can extract features and stacks images
    according to the class folder, also responsible for creation of labels to
    the respective images being processed
    :param image_path: path of the image to be processed
    :param img_folders: folder name where the images are found
    :param mapper: label to numerical encoder
    :return: returns an array with processed image and its labels
    """

    processed_imagestack = []
    label_class = []
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg"):
            img = image.load_img(os.path.join(image_path,filename), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            processed_imagestack.append(x)
            label_class.append(mapper[img_folders])
    return processed_imagestack, label_class


def get_classnames(path,item):
    """
    fetches the class names in a particular folder
    :param path: folder path
    :param item: folder name
    :return: a list of class names
    """
    img_path = os.path.join(path,item)
    return [j for j in os.listdir(img_path)]


def concate_processed_img(path, item, mapper):
    """
    vertical stacking of processed images per folder
    :param path:  path of the folder
    :param item: folder name
    :param mapper: label to numerical encoder
    :return: returns an array of vertical stacked processed images and its labels
    """
    image_stack=[]
    encoded_labels = []
    img_path = os.path.join(path,item)
    for img_folders in os.listdir(img_path):
        folder_iteration_path = os.path.join(img_path,img_folders)
        processed_imgs, labels = process_images_labels(folder_iteration_path, img_folders, mapper)
        encoded_labels += labels
        vstack_img = np.vstack(processed_imgs)
        image_stack.append(vstack_img)
    image_vstack = np.vstack(image_stack)
    return image_vstack, encoded_labels


def extract_feature(img_vstack, model):
    """
    Extracts features from the processed images using pretrained resnet50 model
    :param img_vstack: processed image stack
    :param model: resnet model
    :return: extracted features
    """
    with tf.device('/device:GPU:0'):
        features = model.predict(img_vstack, batch_size= 1) #, batch_size= 50
        features = features.reshape((features.shape[0], 2048))
    return features


def save_files(features, labels, folder_name):
    """
    Saves the feature files as .npy alog with its labels
    :param features: feature file
    :param labels: label file
    :param folder_name: folder name
    :return: None
    """
    np.save(os.path.join(os.getcwd(), "features_"+folder_name+".npy" ), features)
    np.save(os.path.join(os.getcwd(), "labels_"+folder_name+".npy" ), labels)


def process_flow(data_path,folders, model):
    """
    carries out all the above described functions in a flow
    :param data_path: test and train folder path
    :param folders: test or train folder
    :param model: resnet50 model
    :return: none
    """
    print(f'begin processing {folders.upper()} folder..../~')
    classes = get_classnames(data_path,folders)
    label_mapper = {label:encoding for encoding,label in enumerate(classes)}
    concate, labels = concate_processed_img(data_path, folders, label_mapper)
    feature_file = extract_feature(concate, model)
    save_files(feature_file, labels, folders)
    print(f'extracting {folders.upper()} complete..../~')


def Logistic_regrr(Xtrain, Xtest, Ytrain, Ytest):
    """
    trains LR model and also predicts the test cases
    saves the trained model for future use
    :param Xtrain: file
    :param Xtest: file
    :param Ytrain: file
    :param Ytest: file
    :return: prediction results along with accuracy score
    """
    print("Processing LR model")
    LR_model = LogisticRegression(max_iter = 750)
    LR_model.fit(Xtrain, Ytrain)
    pickle.dump(LR_model, open('LRmodel.sav', 'wb'))
    load_LR = pickle.load(open('LRmodel.sav', 'rb'))
    accuracy = load_LR.score(Xtest, Ytest)
    Ypred = load_LR.predict(Xtest)
    return Ypred, accuracy


def MLP_clf(Xtrain, Xtest, Ytrain, Ytest):
    """
      trains MLP model and also predicts the test cases
      saves the trained model for future use
      :param Xtrain: file
      :param Xtest: file
      :param Ytrain: file
      :param Ytest: file
      :return: prediction results along with accuracy score
      """

    print("Processing MLP model")
    mlp_clf = MLPClassifier(hidden_layer_sizes=(2048, 1000, 500, 268),
                        max_iter = 100,activation = 'relu',
                        solver = 'adam')

    mlp_clf.fit(Xtrain, Ytrain)
    pickle.dump(mlp_clf, open('MLPmodel.sav', 'wb'))
    load_MLP = pickle.load(open('MLPmodel.sav', 'rb'))
    Ypred = load_MLP.predict(Xtest)
    accuracy = accuracy_score(Ytest, Ypred)
    return Ypred, accuracy


def get_topten_confusionmatrix(Ypred, Ytest):
    """
    fetches top ten confused classes for model prediction
    :param Ypred: prediction result
    :param Ytest: test data labels
    :return: return the top ten ordered dictionary
    """
    cm = confusion_matrix(Ytest, Ypred)
    cmatrix_dictionary = {}
    for i in range(len(cm)):
        rsum = sum(cm[i][j] for j in range(len(cm[i])) if i!=j)
        cmatrix_dictionary[i] = rsum
    sorted_final_dict = (sorted(cmatrix_dictionary.items(), key = lambda kv:(kv[1], kv[0])))
    return sorted_final_dict[-10:]


def report_topten(Ytest, Ypred, last_ten, data_path):
    """
    Prints the report for each classification model along with top ten miss classified classes report
    :param Ytest: test data
    :param Ypred: prediction result
    :param last_ten: top ten miss classified classes
    :param data_path: data path of the dataset
    :return:
    """
    precision,recall,fscore,support=score(Ytest,Ypred,average='macro')
    print('Precision : {}'.format(precision))
    print( 'Recall    : {}'.format(recall))
    print( 'F-score   : {}'.format(fscore))
    print('Support   : {}'.format(support))
    print("--------------------------------------------------------")
    print("Top ten confused classes")
    report = classification_report(Ytest, Ypred, output_dict=True )
    classes = get_classnames(data_path, 'test')
    label_mapper = {encoding:label for encoding,label in enumerate(classes)}
    for i in last_ten:
        print(label_mapper[i[0]], " : ", report[str(i[0])])


def main():
    data_path = os.path.join(os.getcwd(), 'ReducedSetForAssignment')
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for folders in os.listdir(data_path):
        if folders.endswith(".zip"):
            continue
        elif folders == 'test':
            process_flow(data_path, folders, model)
        elif folders == 'train':
            process_flow(data_path, folders, model)

    Xtrain = np.load("features_train.npy")
    Xtest = np.load("features_test.npy")
    Ytrain = np.load("labels_train.npy")
    Ytest = np.load("labels_test.npy")

    #train and report logistic regression classifier:
    Ypred_LR, accuracy_LR = Logistic_regrr(Xtrain, Xtest, Ytrain, Ytest)
    print(f'Logistic regression classifier accuracy: {accuracy_LR}')
    topten_LR = get_topten_confusionmatrix(Ypred_LR, Ytest)
    report_topten(Ytest, Ypred_LR, topten_LR, data_path)
    print()
    #train and report MLP classifier:
    Ypred_MLP, accuracy_MLP = MLP_clf(Xtrain, Xtest, Ytrain, Ytest)
    print(f'MLP classifier accuracy: {accuracy_MLP}')
    topten_MLP = get_topten_confusionmatrix(Ypred_MLP, Ytest)
    report_topten(Ytest, Ypred_MLP, topten_MLP, data_path)



if __name__ == '__main__':
    main()

