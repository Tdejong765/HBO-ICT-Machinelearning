import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras import models, layers
from keras.layers import Flatten



# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    # YOUR CODE HERE

    # colormap: binary
    plt.imshow(img, plt.cm.binary)

    # titel van image is label 
    plt.title(label)

    # show de plot
    plt.show()


# OPGAVE 1b
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    # YOUR CODE HERE

    div = X / np.amax(X)

    return div
    

# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    # YOUR CODE HERE

    model = models.Sequential()  # keras.Sequential()
    model.add(layers.Dense(784, input_shape=(28, 28)))
    model.add(Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    
    # YOUR CODE HERE


    return tf.math.confusion_matrix(labels, pred)

    
# OPGAVE 2b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
 
    print("#### conf ####")
    print(conf)
    # YOUR CODE HERE
    for i in range(len(labels)):
        
                #schuine true positive waarde
                TP = np.diag(conf)
                TP = TP[i]

                #false positive waardes van Tshirt
                FP = conf[i].sum(axis=0) -np.diag(conf)
                FP = sum(FP)

                #false negative waardes van Tshirt
                c = conf[:, i]
                FN = c.sum(axis=0) - np.diag(conf)
                FN = sum(FN)
            
                #true negative waardes van Tshirt
                TN = conf[i].sum(axis=0) - (FP + FN + TP)
                TN = TN.sum()

                if i == 0:
                    Tshirt = (labels[i], TP, FP, FN, TN)
                if i == 1:
                    Broek = (labels[i], TP, FP, FN, TN)
                if i == 2:
                    Pullover = (labels[i], TP, FP, FN, TN)
                if i == 3:
                    Jurk = (labels[i], TP, FP, FN, TN)
                if i == 4:
                    Jas = (labels[i], TP, FP, FN, TN)
                if i == 5:
                    Sandalen = (labels[i], TP, FP, FN, TN)
                if i == 6:
                    Shirt = (labels[i], TP, FP, FN, TN)
                if i == 7:
                    Sneaker = (labels[i], TP, FP, FN, TN)
                if i == 8:
                    Tas = (labels[i], TP, FP, FN, TN)
                if i == 9:
                    Laars = (labels[i], TP, FP, FN, TN)


    confusion = [Tshirt, Broek, Pullover, Jurk, Jas, Sandalen, Shirt, Sneaker, Tas, Laars]
    return confusion




# OPGAVE 2c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).
    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE

    #TP = 1, FP = 2, FN = 3, TN = 4
   
    for i in range(len(metrics)):
            TPR = metrics[i][1] / (metrics[i][1] + metrics[i][3])
            PPV = metrics[i][1] / (metrics[i][1] + metrics[i][2])
            TNR = metrics[i][4] / (metrics[i][4] + metrics[i][2])
            FPR = metrics[i][2] / (metrics[i][2] + metrics[i][4])

    print("### TPR ###")
    print(TPR)
    
    print("### PPV ###")
    print(PPV)
     
    print("### TNR ###")
    print(TNR)

    print("### FPR ###")
    print(FPR)
      
            
    # BEREKEN HIERONDER DE JUISTE METRIEKEN EN RETOURNEER DIE 
    # ALS EEN DICTIONARY

    rv = {'tpr':TPR, 'ppv':PPV, 'tnr':TNR, 'fpr':FPR }
    return rv
