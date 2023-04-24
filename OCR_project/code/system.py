"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
import data
from PIL import Image



def reduce_dimensions(feature_vectors_full, model):
          '''

          Because of the limitations to the json file size only the first 1150 features are used to get the pcas
          of the train data.
          the first 1150 features are stored in the model dictionary with key labeled 14395
          the divergence of each feature  of the  pca are calculated and then summed for each
           feature and then the features with the highest
          divergences are picked
          the features for the train data and the indexes of the best pcas are then used to reduce the dimensionality
          of the test data

          parameters
          feature_vectors
          model: contains a new key 14395 indicating the shape of the train data and best containing the indexes
          of the 10 best features calculated from the divergences



          '''


          #if exception because key is not in dictionary. When the train data is reduced,
          # key and value containing train features does not exist in dictionary

          try:  np.array(model['14395'])

          except KeyError:


                  #creates a dictionary containing all the labels
                  label = np.array(model["labels_train"])
                  unique = []
                  for i in range(len(label)):
                      if label[i] not in unique:
                          unique.append(label[i])
                      # dictionary label:index
                  u = {}
                  for i in range(len(unique)):
                      u[unique[i]] = int(i)


                  #first 1150 feature vectors
                  data =feature_vectors_full[:,:1150]
                  model[feature_vectors_full.shape[0]] = data.tolist()



                  #gets the first 20 pcas
                  covx = np.cov(data, rowvar=0)
                  N = covx.shape[0]
                  w, v = scipy.linalg.eigh(covx, eigvals=(N - 20, N - 1))
                  v = np.fliplr(v)
                  #print(v)
                  #print(w)
                  pca = np.dot((data - np.mean(data)), v)
                  # inserts the first 1350 train feature vectors into model


                  #converts each label in label list into their corresponding number from the u dictionary
                  #where each unique label has a number from 1-56
                  #the labels in the label list are then converted to that number
                  ldx = list(label.copy())
                  for i in range(len(ldx)):
                      if ldx[i] in u.keys():
                          ldx[i] = u[ldx[i]]
                  ldx = np.array(ldx)



                  # finds all the rows corresponding to their label number and the computes the divergence with
                  # all the rows with another label
                  np.seterr(divide='raise',invalid='raise')
                  div = []
                  for i in range(57):
                      for j in range(57):
                          if i != j:
                            try:
                              div.append(list(divergence(pca[ldx[0:] == i, :], pca[ldx[0:] == j, :])))
                             # print(i)
                             # print(pca[ldx[0:] == i, :])
                            except FloatingPointError:
                                continue


                  #convert to array
                  dive = np.array(div)

                  #sum divergence of each feature
                  diver = np.sum(dive, axis=0)

                  #appends  tuple of divergence and index in a new list and then sorts the list.
                  diverg = []
                  for i in range(len(diver)):
                      diverg.append((i, diver[i]))
                  diverg.sort(key=lambda tup: tup[1])

                  #finds index of the top ten features with the highest divergences

                  best = []
                  for i in diverg:
                      best.append(i[0])
                  best = best[-10::]

                  #extracts the best pca features
                  tentrain = pca[:, best]

                  #puts the index into a dictionary
                  model['best']= best


                  return tentrain




          best =model['best']
          pcatrain = model['14395']
          #calculate the covariance matrix of the train data
          covx = np.cov(pcatrain, rowvar=0)
          N=covx.shape[0]
          #gets the first 1350 features of the test data
          test = feature_vectors_full[:, :np.array(pcatrain).shape[1]]
          #calculates the top 20 pcas
          w, v = scipy.linalg.eigh(covx, eigvals=(N - 20, N - 1))
          v = np.fliplr(v)
          pca = np.dot((test- np.mean(pcatrain)), v)
          #gets the 10 best pcas from using the 10 best from the train data
          tentest = pca[:,best]
          # print(pca.shape)


          return tentest






def divergence(c1, c2):
    """compute a vector of 1-D divergences
       Taken from cocalc lab

       class1 - data matrix for class 1, each row is a sample
       class2 - data matrix for class 2

       returns: d12 - a vector of 1-D divergence scores

       This function was taken from cocalc assignment
       """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(c1, axis=0)
    m2 = np.mean(c2, axis=0)
    v1 = np.var(c1, axis=0)
    v2 = np.var(c2, axis=0)
   # print(v1)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12





def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.


def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """

    print("Reading data")
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print("Extracting features from training data")
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)


    model_data = dict()
    model_data["labels_train"] = labels_train.tolist()
    model_data["bbox_size"] = bbox_size

    #stores the corncob txt file into a list using stoplist function and then into 'stop' key in model
    #dictionary
    try:
        model_data['stop']


    except KeyError:
        model_data['stop']= stoplist("corncob_lowercase.txt")


    print("Reducing to 10 dimensions")


    fvectors_train= reduce_dimensions(fvectors_train_full, model_data)

    model_data["fvectors_train"] = fvectors_train.tolist()




    return model_data



def load_test_page(page_name, model):
    import copy
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model["bbox_size"]
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced= reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """
    knn algorithm used to classify  data after getting cosine distances

    parameters:
    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])
    fvectors_train = np.array(model["fvectors_train"])
    #calculates the cosine distance between each feature in the test and train data
    x= np.dot(page,fvectors_train.transpose())
    test = np.sqrt(np.sum(page * page, axis=1))
    train = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(test,train.transpose())
    knearest= knn(5,dist,labels_train)

    return knearest


def knn(num,dist,labels):
    '''

    The function takes the list of distances and then finds the largest distance in the list and appends
    the index of the distance the into a new list.
    It then replaces the distance with 0 and finds the next biggest distance. it runs while the length
    of the index list is smaller than the num/k. Since the index of the distances are the same as the labels
    it could be used to retrieve the k nearest labels which have the k largest distances. the mode of label
    list is then taken to find the highest frequency label out of the top k

    parameter:
    num: the number of nearest neighbours similar to feature before classifying it
    dist: the cosine distance of each feature to the test data
    labels: list of labels of the train data


    '''

    labelt=[]
    for i in range(len(dist)):
      ls=[]
      a =dist[i].copy()
      while len(ls)<num:

          n = np.argmax(a)
          ls.append(n)
          a[n]=0
      for l in range(len(ls)):
          ls[l]=labels[ls[l]]

      from collections import Counter
      # print(N)
      c = Counter(ls)

      #  print(c)
     # print(c)

      d = [k for k, v in c.items() if v == c.most_common(1)[0][1]]
     # print(d)
      labelt.append(d[0])


    return labelt


def stoplist(sl):
    '''
    turns txt file into list of  words
    parameters:
    sl:txt file
    '''
    f = open(sl, "r")
    st = []
    for line in f:
        st.append(line.strip())
    return st






def correct_errors(page, labels, bboxes, model):
    """
    function extracts x axis values from the bboxes array and converts into a list of tuples with each tuple having
    the same index as the label list
    then the label list is sliced where the difference between first x value of the next tuple and the second value of
    the previous tuple is greater than 7 and then appended into a new word list
    new  word list is iterated through and for each element of new word list, word is joined back into a string
    and then checked to see if it exists in the stoplist. If it does not exist in the stoplist, iterate through each letter of word
    and consecutively converting each character to its ascii value starting from the first letter and iterating
    through all ascii values  and converting the ascii value back to the character at each step
    and replacing the original character with the new character and then checking to see if word is in the stoplist, if it is then
    convert word back into a list and replace original list of letters with new word list otherwise continue on until
    the last letter and  last ascii value  where the word list will remain the same

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage

    """

    #creates a list of zeros with the lenegth of the labels
    c = list(labels)
    lz = [0] * len(c)
    #converts list to list of tuples of the x axis values
    for i in range(len(lz)):
        lz[i]=(bboxes[i][0],bboxes[i][2])

    word=[]
    ww=[]
    count=0
    #slices the label list c where the first value of the next tuple if greater that the 2nd value of the previous tuple.
    # tuple index in list of x axis tuples matches index of label list.
    #label list will contain list of word slices of actual words estimates by their x axis distance
    for i in range(len(lz)-1):

        end=i+1
        start=i
        #w=''
        count+=1

        if lz[end][0]-lz[start][1] >=7:

            word.append(c[end-count:end])
            count=0
        elif  end==len(lz)-1:
            word.append(c[end - count:end+1])

            count=0
        else:
            continue

    #gets the stop-list from model dictionary
    stop=model['stop']

    num=1
    count=0

    for i in range(len(word)):
      #start = 0
      #turns list of letters into a word
      w=word[i]
      we = ''.join(w)
      wc=list(we)
      #lowers the first letter of a word to find out if its in stop list because stop list is lowercase
      wc[0]=wc[0].lower()
      wc=''.join(wc)
      #if word not in stop and word doesnt contain any punctuation
      if we not in stop and we.isalpha() and wc not in stop:

          #converts letter to its ascii value and then increases it by 1, converts it back to a letter and replaces
          #original letter in word with new letter.
          #checks if word is in stoplist is its in stoplist then replace original letter list with new letter list
          #else continue iterating until the last ascii value 122
          for l in range(len(we)):
              copy=we[l]
              num=ord(we[l])

              for j in range(num,123):
                  ww=we.replace(we[l],chr(j))

                  if ww not in stop:
                    ww=we.replace(we[l],copy)

                  elif ww not in stop and j ==122:
                    word[i]=list(ww)

                  elif ww in stop:

                    #print(ww,'found')

                     #print(word[i])
                     if ord(we[0]) < 90:
                       #if first letter is capital, convert first letter back to capital
                       wp=list(ww)
                       wp[0]=ww[0].upper()
                       word[i]=wp
                       #print(list(wp),'found')
                     else:
                       word[i]=list(ww)
                       #print(list(ww), 'found')

                     count+=1
                     break

      else:
          continue

    #flattens list
    l = [item for sublist in word for item in sublist]
    label = np.array(l)


    return label

