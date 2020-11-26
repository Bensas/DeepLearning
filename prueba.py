import numpy as np
from random import random
from random import randint
prob_of_change = 0.99
font = [
   [0, 1, 1, 1, 0, 0, 0],
   [0, 1, 1, 1, 0, 0, 0]]
quantity=1
salt_and_pepper=0.5
size=len(font)*len(font[0])
how_much_salt = int(np.ceil(quantity * size * salt_and_pepper))
i = 0
while( i < how_much_salt):
    x = randint(0, len(font)-1)
    y = randint(0, len(font[0])-1)
    font[x][y] = 1 
    i = i + 1

how_much_pepper = int(np.ceil(quantity * size * salt_and_pepper))
i = 0
while( i < how_much_pepper):
    x = randint(0, len(font)-1)
    y = randint(0, len(font[0])-1)
    font[x][y] = 0
    i = i + 1

print(font)