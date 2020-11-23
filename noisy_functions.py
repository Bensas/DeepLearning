import numpy as np
from random import random
from random import randint

def noisy_function_lighter(fonts, prob_of_change):
    simbols = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f]
    noisy_fonts = [0] * len(fonts)
    for index in fonts:
        if(random() < prob_of_change):
            noisy_fonts[index] = simbols[random.randint(0,len(simbols)-1)]
        else:
            noisy_fonts[index] = fonts[index]
    return noisy_fonts

def noisy_function_heavier(fonts, prob_of_change):
    noisy_fonts = [0] * len(fonts)
    for index in fonts:
        if(random() < prob_of_change):
            if (noisy_fonts[index] < 1):
                noisy_fonts[index] = noisy_fonts[index] + 1
            else:
                noisy_fonts[index] = noisy_fonts[index] - 1
        else:
            noisy_fonts[index] = fonts[index]
    return noisy_fonts

def hexa_to_binary(fonts):
    matrix = [0] * len(fonts)
    index_matrix = 0
    while(index_matrix < len(fonts)):
        font_in_fonts = fonts[index_matrix]
        answer = [0] * len(font_in_fonts) * 5
        index_answer = 0
        index_array = 0
        while(index_answer < len(font_in_fonts)):
            integer = str(bin(int(str(font_in_fonts[index_answer]), 10)))
            array = integer[2:]
            if(len(array) < 5):
                while(index_array < (5 - len(array))):
                    answer[index_answer*5 + index_array] = 0
                    index_array = index_array + 1
            while(index_array < 5):
                answer[(index_answer*5) + index_array] = int(array[index_array - (5 - len(array))])
                index_array = index_array + 1
            index_answer = index_answer + 1
            index_array = 0
        matrix[index_matrix] = answer
        index_matrix = index_matrix + 1
    return matrix