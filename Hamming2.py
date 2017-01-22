import sys
import math
import pickle
import numpy
import copy
import random
from numpy import matrix, identity
from bitarray import bitarray, bits2bytes

DEBUG = False
CODEWORD_SIZE = 8
INFO_BITS = 4


class HammingEncoder:
    def __init__(self, data: bitarray):
        self.raw_data = data
        self.codeword_size = CODEWORD_SIZE
        self.info_bits = INFO_BITS
        self.G = matrix([[1, 0, 0, 0, 0, 1, 1, 1],
                         [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 1, 1, 1, 1, 0]])
        self.encoded_data = bitarray()

    def _get_next_word(self):
        i = 0
        while True:
            data_chunk = self.raw_data[i:i + self.info_bits]
            chunk_length = data_chunk.length()
            if chunk_length == 0:
                break
            i += data_chunk.length()
            yield data_chunk

    def encode(self):
        print('G matrix = ')
        print(self.G)

        for chunk in self._get_next_word():
            vector_x = matrix(chunk, dtype=int)
            encoded_chunk = vector_x * self.G % 2
            encoded_chunk_bitarray = bitarray()
            encoded_chunk_bitarray.extend(encoded_chunk.A1)

            print('X =', vector_x.A1)
            print('Encoded = ', encoded_chunk.A1)

            self.encoded_data += encoded_chunk_bitarray
        return self.encoded_data


class HammingDecoder:
    def __init__(self, encoded_data: bitarray):
        self.codeword_size = CODEWORD_SIZE
        self.info_bits = INFO_BITS
        self.encoded_data = encoded_data
        self.decoded_data = bitarray()
        self.H = matrix([[0, 1, 1, 1],
                         [1, 0, 1, 1],
                         [1, 1, 0, 1],
                         [1, 1, 1, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        self.syndrome_dict = {'0111': 0, '1011': 1, '1101': 2, '1110': 3, '1000': 4, '0100': 5, '0010': 6, '0001': 7, '0000': -1}

    def _get_next_word(self):
        i = 0
        while True:
            data_chunk = self.encoded_data[i:i + self.codeword_size]
            chunk_length = data_chunk.length()
            if chunk_length == 0:
                break
            i += data_chunk.length()
            yield data_chunk

    def decode(self):
        print('H = ')
        print(self.H)
        print('Syndrome dictionary = ', self.syndrome_dict)

        for chunk in self._get_next_word():
            vector_y = matrix(chunk, dtype=int)
            syndrome = ''.join(map(str, ((vector_y * self.H) % 2).A1))

            print(' Y = ', vector_y)
            print('Syndrome = ', syndrome)

            try:
                error_position = self.syndrome_dict[syndrome]
            except:
                print('Two Errors in word')
                exit()

            if error_position != -1:
                vector_y[0, error_position] = not vector_y[0, error_position]
                print('Error in position ', error_position)
                print('True Y = ', vector_y.A1)
            print('Z = ', vector_y.A1[:self.info_bits])
            self.decoded_data.extend(vector_y.A1[:self.info_bits])
        return self.decoded_data


def hamming_encoder():
    with open('files\origin', encoding="utf-8") as input_file:
        input_text = input_file.read()

    print(input_text)

    bit_data = bitarray()
    bit_data.fromstring(input_text)

    HE = HammingEncoder(data=bit_data)
    hamming_encoded_data = HE.encode()

    print(hamming_encoded_data)

    with open('files\encoded_data', mode='wb') as encoded_file:
        hamming_encoded_data.tofile(encoded_file)


def hamming_decoder():
    with open('files\encoded_data', 'rb') as encoded_file:
        encoded_text_bitarray = bitarray()
        encoded_text_bitarray.fromfile(encoded_file)

    HD = HammingDecoder(encoded_data=encoded_text_bitarray)
    decoded_data = HD.decode()

    print(decoded_data)

    text_data = decoded_data.tostring()
    print(text_data)

    with open('files\decoded_data', 'w', encoding="utf-8") as decoded_file:
        decoded_file.write(text_data)


def add_errors():
    with open('files\encoded_data', 'rb') as encoded_file:
        encoded_text_bitarray = bitarray()
        encoded_text_bitarray.fromfile(encoded_file)

    print("Corrupted bits: ", end="")
    for i in range(8, len(encoded_text_bitarray), CODEWORD_SIZE):
        e = random.random()
        if e <= 0.02:
            encoded_text_bitarray[i] = not encoded_text_bitarray[i]
            print(str(i), end=" ")

    with open('files\encoded_data', 'wb') as encoded_file:
        encoded_text_bitarray.tofile(encoded_file)


def add_two_errors_in_one_word():
    with open('files\encoded_data', 'rb') as encoded_file:
        data = bitarray()
        data.fromfile(encoded_file)

    flag = True
    print("Corrupted bits: ", end="")
    errors = 0
    for i in range(8, len(data), CODEWORD_SIZE):
        e = random.random()
        p = random.randint(0, 7)
        p1 = random.randint(0, 7)
        if e <= 0.02:
            data[i + p] = not data[i + p]
            data[i + p1] = not data[i + p1]
            errors += 2
            print(str(i + p), end=" ")
            print(str(i + p1), end=" ")

    with open('files\encoded_data', 'wb') as encoded_file:
        data.tofile(encoded_file)

if __name__ == '__main__':
    hamming_encoder()
    add_two_errors_in_one_word()
    hamming_decoder()
    print("Hamming end.")
