from audioop import bias
from tkinter import E
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    x = np.reshape(nrVector.T, (20, 20), order='F')
    plt.matshow(x)
    plt.show()
    

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
    calc = 1+np.exp(-z)
    sig = 1 / calc
    return sig


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m
    #YOUR CODE HERE
    

    #matrix van alleen cijfers 0 tot 10 maken
    cols = [i[0] if i[0] != 10 else 0 for i in y]

    #grote van matrix bepalen
    rows = [i for i in range(m)]

    #1 op de juiste positie zetten
    data = [1 for _ in range(m)]

    #breedte van matrix bepalen
    width = max(cols) # arrays zijn zero-based

    #Ijle matrix maken
    y_vec = csr_matrix((data, (rows, cols)), shape=(len(rows), width+1)).toarray()

    print("######## ijle matrix ###########")
    print(y_vec)

    return y_vec

    
# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.


    ### inputlayer ###
    # 1. 
    #size van matrix
    size = X.shape[0]
    #enen toevoegen aan de matrix X
    biasInput = np.c_[np.ones(size), X]
    

    ###  hiddenlayer ###
    # 2. 
    #Matrix met eerste gewichten
    weightHidden = np.dot(Theta1, biasInput.T) 
    #Sigmoid van deze matrix
    activationHidden = sigmoid(weightHidden)
    # size van matrix hiddenlayer
    size2 = activationHidden.shape[1]
    #Enen toevoegen aan matrix (bias)
    biasHidden = np.c_[np.ones(size2), activationHidden.T]

    ### outputlayer ### 
    # 3.
    #Gewichten van deze matrix (Theta2)
    weightOutput = np.dot(Theta2, biasHidden.T)
    # sigmoid op output
    activationOutput = sigmoid(weightOutput)

    #sg transposen
    return activationOutput.T



# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 


    # M aantal observaties
    m = y.shape[0]

    # Matrix grootte van oberservaties met actuele waarde
    y_mat = get_y_matrix(y, m)

    # Hypothese (voorspelling)
    h = predict_number(Theta1, Theta2, X)

    # Matrix actuele waarde * de log van de hypothese (kost als y=1)
    yLogH = y_mat * np.log(h)

    # Matrix actuele waarde * de log van de hypothese (kost als y=0)
    yLogHMin = (1 - y_mat) * np.log(1 - h)

    # Som van de kosten als y=0 & y=1
    sumOfK = yLogH + yLogHMin

    # Som van de observaties
    sumOfM = np.sum(sumOfK)

    #Som van observaties gedeeld door aantal observaties = uiteindelijke kost
    cost = -sumOfM / m
    return cost



# ==== OPGAVE 3a ====
def sigmoid_gradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.

    #sigmoid
    sig = sigmoid(z)

    #sigmoid * 1 - zichzelf doen
    calc = (sig*(1-sig))

    #grote hiervan returnen
    return calc[0]

# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y): 
    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    #size van m
    m = X.shape[0] #niet meer een voorbeeldwaarde


    ################## FORWARDPROPAGATION ###############
    ### inputlayer ###
    biasInput = np.c_[np.ones((len(X), 1)), X] 

    ### hiddenlayer ###
    weightHidden = np.dot(biasInput, Theta1.T) 
    activationHidden = sigmoid(weightHidden) 
    
    ### outputlayer
    biasOutput = np.c_[np.ones((len(activationHidden), 1)), activationHidden] 
    weightOutput = np.dot(biasOutput, Theta2.T)  
    activationOutput = sigmoid(weightOutput) 


    ################## BACKWARDPROPAGATION ###############
    # pak de ijle martix. Zodat je elke observatie individueel hebt.
    yMatrix = get_y_matrix(y,m)
    
    # bereken de foutmarge van de outputlayer: verschil tussen de output en de observatie (Inproduct van de fout)
    deltaOutput = activationOutput - yMatrix    


    # Foutmarge keer Theta2 (weightsHidden). 
    deltaHidden = Theta2.T.dot(deltaOutput.T)
    # haal bias weg
    deltaHidden = np.delete(deltaHidden,0,0) 
    # individuele bijdrage aan de totale fout = inproduct van de fout * afgeleide van de sigmoïdefunctie.
    deltaHidden = deltaHidden.T * sigmoid_gradient(weightHidden)
    print(deltaHidden)



    #Updaten beide Delta matrixen, verschillen tussen voorspelde en actuele waarde
    Delta2 += np.dot(biasInput.T,deltaHidden).T
    Delta3 += np.dot(biasOutput.T,deltaOutput).T

    #Updaten beide Theta matrixen
    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    return Delta2_grad, Delta3_grad

