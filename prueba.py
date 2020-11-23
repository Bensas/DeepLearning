fonts = [0x00,0x01,0x02,0x1a,0x01f]
answer = [0] * len(fonts) * 5
index_answer = 0
index_array = 0
while(index_answer < len(fonts)):
    integer = str(bin(int(str(fonts[index_answer]), 10)))
    array = integer[2:]
    if(len(array) < 5):
        while(index_array < (5 - len(array))):
            answer[index_answer*5 + index_array] = '0'
            index_array = index_array + 1
    while(index_array < 5):
        answer[(index_answer*5) + index_array] = array[index_array - (5 - len(array))]
        index_array = index_array + 1
    index_answer = index_answer + 1
    index_array = 0
return answer
