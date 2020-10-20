"""
Sukurkite paprasto klasifikatoriaus (Perceptrono) išėjimui apskaičiuoti skirtą programą.
Klasifikatorius turi skirstyti objektus į dvi klases, pagal du požymius.
Išėjimo skaičiavimas atliekamas pagal formulę: y = 1, kai x1*w1 + x2*w2 + b > 0; y = -1, kai x1*w1 + x2*w2 + b <= 0;
čia w1, w2 ir b parametrai, kurie turi būti sugeneruojami naudojant atsitiktinių skaičių generatorių
(MATLAB pvz.: w1 = randn(1);) pirmąją programos veikimo iteraciją ir vėliau atnaujinami mokymo algoritmu;
x1 ir x2 yra objektų požymiai, apskaičiuoti Matlab funkcijomis, esančiomis paruoštame kodo ruošinyje arba
Data.txt faile (kiekvienoje eilutėje yra toks duomenų formatas: požymis1, požymis2, norimas_atsakymas),
jei ketinate naudoti ne Matlab..

"""
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

# read Data file
text_file = open("Data.txt", "r")
c = StringIO(u"0 1\n2 3")
x1, x2, T = np.loadtxt(text_file, delimiter=',', usecols=(0, 1, 2), unpack=True)

# Sugeneruojami random koeficientai w1 w2 b

data_length = len(x1)
w1 = np.random.rand(data_length, )
w2 = np.random.rand(data_length, )
b = 0  # arba galima w = np.zeros((1,))


def OutputandLossCalculate(w1, w2, b, x1, x2):
    """
    FUunkcija paskaičiuoja parceptrono išėjimą

    Arguments:
        w1 -- x1 įėjimo svoris
        w2 -- x2 įėjimo svoris
        b -- parceptrono svoris
        x1 -- x1 iejimo duomenys
        x2 -- x2 iejimo duomenys
        T -- tikrasis atsakymas

    Returns:
        v -- išejimo reiksmė
        y -- priskirta isejimo reiksme
        e -- paskaičiuota klaida
   """

    v = x1 * w1 + x2 * w2 + b
    y = np.zeros(data_length, )

    for x in range(data_length):
        if v[x] > 0:
            y[x] = 1
        else:
            y[x] = -1

    # Randame e

    e = T - y

    return v, y, e


def optimize(w1, w2, b, x1, x2, e, num_iterations, learning_rate):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w1 -- x1 svoris
    w2 -- x2 svoris
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- iteraciju skaičius naudojamas apmokinimui
    learning_rate -- mokymosi žingsinis

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.


   """

    for i in range(num_iterations):
        w1 = w1 + learning_rate * e * x1
        w2 = w2 + learning_rate * e * x2

    b = b + learning_rate * e

    return w1, w2, b


plt.plot(x1,  'ro', x2, '*')
plt.show()


v, y, e = OutputandLossCalculate(w1, w2, b, x1, x2)

proc = 100 - np.mean((np.abs(np.sum(e))/2) / data_length) * 100

print("Tikslumas pries apmokymo  %", proc)

learning_rate = 0.5
num_iterations = 5

w1, w2, b = optimize(w1, w2, b, x1, x2, e, num_iterations, learning_rate)


#palyginimui

v, y, e = OutputandLossCalculate(w1, w2, b, x1, x2)

proc = 100 - np.mean((np.abs(np.sum(e))/2) / data_length) * 100

print("Tikslumas po apmokymo  %", proc)

print(e)