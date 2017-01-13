import sys
import math
import pickle
import numpy
from numpy import matrix, identity
from bitarray import bitarray, bits2bytes

USELESS_DATA_AMOUNT = math.ceil(math.log2(3))


class Help:
    @staticmethod
    def write_to_file(EncodedData: bitarray, Filename: str):
        UselessEndBits = bits2bytes(EncodedData.length()) * 8 - EncodedData.length() - 3

        for i in range(0, 3):
            EncodedData.insert(0, (UselessEndBits >> (2 - i)) & 1)

        with open(Filename, mode="wb") as encoded_file:
            EncodedData.tofile(encoded_file)

    @staticmethod
    def read_from_file(Filename: str) -> bitarray:
        with open(Filename, "rb") as encoded_file:
            EncodedTextBitarray = bitarray()
            EncodedTextBitarray.fromfile(encoded_file)

        UselessEndBits = 0
        for i in range(0, 3):
            UselessEndBits += EncodedTextBitarray.pop(0) << i

        for i in range(UselessEndBits):
            EncodedTextBitarray.pop()

        return EncodedTextBitarray

    @staticmethod
    def get_parity_bits_matrix() -> matrix:
        return matrix([[1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]])

    @staticmethod
    def get_generator_matrix() -> matrix:
        identity_matrix = identity(4, dtype=int)
        parity_bits_matrix = Help.get_parity_bits_matrix()
        return numpy.append(identity_matrix, parity_bits_matrix, axis=1)

    @staticmethod
    def get_parity_check_matrix() -> matrix:
        identity_matrix = identity(3, dtype=int)
        parity_bits_matrix = Help.get_parity_bits_matrix()
        return numpy.append(parity_bits_matrix.T, identity_matrix, axis=1)

    @staticmethod
    def get_syndrome_dict(parity_check_matrix: matrix) -> dict:
        syndrome_dict = {}
        for i in range(-1, 7):
            error_code = matrix([0] * 7, dtype=int)
            if i != -1:
                error_code[0, i] = 1
            syndrome = ((parity_check_matrix * error_code.T) % 2).A1
            syndrome_dict["".join(map(str, syndrome))] = i
        return syndrome_dict


class HuffmanNode:
    def __init__(self, char=None, frequency=None, lchild=None, rchild=None, parent=None):
        self.char = char
        self.frequency = frequency
        self.left_child = lchild
        self.right_child = rchild
        self.parent = parent

    def __repr__(self, *args, **kwargs):
        return "(" + self.char + ", " + str(self.frequency) + ")"


class HuffmanEncoder:
    def __init__(self, text: str):
        self.original_text = text
        self.encoded_text = None
        self.frequency_table = None
        self.huffman_table = None
        self.entropy = 0

    def encode(self) -> bitarray:
        self.frequency_table = self._make_frequency_table(self.original_text)
        self.entropy = self._compute_entropy(self.frequency_table, len(self.original_text))
        huffman_tree = self._make_huffman_tree(self.frequency_table)
        self.huffman_table = self._make_huffman_table(huffman_tree)
        encoded_text = bitarray()
        encoded_text.encode(self.huffman_table, self.original_text)
        return encoded_text

    @staticmethod
    def _make_frequency_table(text: str) -> dict:
        freq_table = {}
        for c in text:
            freq_table[c] = freq_table[c] + 1 if c in freq_table else 1
        return freq_table

    @staticmethod
    def _compute_entropy(freq_table: dict, symbols_count: int) -> float:
        entropy = 0
        for key in freq_table:
            key_frequency = freq_table[key] / symbols_count
            entropy -= key_frequency * math.log2(key_frequency)
        return entropy

    @staticmethod
    def _make_huffman_tree(freq_table: dict) -> HuffmanNode:
        def _pop_first_nodes(nodes: list):
            if len(nodes) > 1:
                _first = nodes.pop(0)
                _second = nodes.pop(0)
                return _first, _second
            else:
                return nodes[0], None

        processing_queue = []
        for key in sorted(freq_table, key=freq_table.get, reverse=True):
            processing_queue.append(HuffmanNode(key, freq_table[key]))
        while True:
            first, second = _pop_first_nodes(processing_queue)
            if not second:
                return first
            parent = HuffmanNode(lchild=first, rchild=second, frequency=first.frequency + second.frequency)
            first.parent = parent
            second.parent = parent
            processing_queue.insert(0, parent)
            processing_queue.sort(key=lambda node: node.frequency)
        return processing_queue.pop()

    @staticmethod
    def _make_huffman_table(huffman_tree_root: HuffmanNode) -> dict:
        def _compute_huffman_code(node: HuffmanNode, codes: dict, bits: bitarray):
            if not node.left_child and not node.right_child:
                codes[node.char] = bits.copy() if bits.length() > 0 else bitarray('0')
                return

            if node.left_child:
                bits.append(False)  # Left branch
                _compute_huffman_code(node.left_child, codes, bits)
                bits.pop()

            if node.right_child:
                bits.append(True)  # Right branch
                _compute_huffman_code(node.right_child, codes, bits)
                bits.pop()

        codes_dict = {}
        _compute_huffman_code(huffman_tree_root, codes_dict, bitarray())
        return codes_dict


class HuffmanDecoder:
    def __init__(self, encoded_data: bitarray, codemap: dict):
        self.encoded_data = encoded_data
        self.codemap = codemap
        self.decoded_text = None

    def decode(self) -> str:
        self.decoded_text = self.encoded_data.decode(self.codemap)
        return self.decoded_text


class HammingEncoder:
    def __init__(self, data: bitarray, codeword_size: int, info_bits: int):
        self.raw_data = data
        self.codeword_size = codeword_size
        self.info_bits = info_bits
        self.generator_matrix = Help.get_generator_matrix()
        self.encoded_data = bitarray()
        self.useless_encoded_bits = 0

    def encode(self) -> bitarray:
        print("Matrix G =")
        print(self.generator_matrix)

        for chunk in self._get_next_chunk():
            vector_x = matrix(chunk, dtype=int)
            encoded_chunk = vector_x * self.generator_matrix % 2
            encoded_chunk_bitarray = bitarray()
            encoded_chunk_bitarray.extend(encoded_chunk.A1)

            print("Vector X =", vector_x.A1)
            print(" Encoded =", encoded_chunk.A1)
            print()

            self.encoded_data += encoded_chunk_bitarray
        return self.encoded_data

    def _get_next_chunk(self):
        i = 0
        while True:
            data_chunk = self.raw_data[i:i + self.info_bits]
            chunk_length = data_chunk.length()
            if chunk_length == 0:
                break
            if chunk_length < self.info_bits:
                self.useless_encoded_bits = self.info_bits - chunk_length
                data_chunk.extend("0" * self.useless_encoded_bits)
            i += data_chunk.length()
            yield data_chunk


class HammingDecoder:
    def __init__(self, encoded_data: bitarray, codeword_size: int, info_bits: int):
        self.codeword_size = codeword_size
        self.info_bits = info_bits
        self.encoded_data = encoded_data
        self.decoded_data = bitarray()
        self.parity_matrix = Help.get_parity_check_matrix()
        self.syndrome_dict = Help.get_syndrome_dict(self.parity_matrix)

    def decode(self) -> bitarray:
        print("Matrix H =")
        print(self.parity_matrix)
        print(self.syndrome_dict)

        for chunk in self._get_next_chunk():
            vector_y = matrix(chunk, dtype=int)
            syndrome = "".join(map(str, ((vector_y * self.parity_matrix.T) % 2).A1))

            print("Vector Y =", vector_y.A1)
            print("Syndrome =", syndrome)

            error_position = self.syndrome_dict[syndrome]
            if error_position != -1:
                vector_y[0, error_position] = not vector_y[0, error_position]

                print("error in position", error_position)
                print(" true y vector = ", vector_y.A1)

            print("Result =", vector_y.A1[:self.info_bits])
            print()
            self.decoded_data.extend(vector_y.A1[:self.info_bits])
        return self.decoded_data

    def _get_next_chunk(self):
        i = 0
        while True:
            data_chunk = self.encoded_data[i:i + self.codeword_size]
            chunk_length = data_chunk.length()
            if chunk_length == 0:
                break
            i += data_chunk.length()
            yield data_chunk


class Menu:
    @staticmethod
    def RunEncoder(InputFileName: str, DestinationFileName: str, CodesTableFileName: str):
        with open(InputFileName, encoding="utf-8") as InputFile:
                InputText = InputFile.read()

        HuffmanEnc = HuffmanEncoder(InputText)
        EncodedData = HuffmanEnc.encode()
        CodesTable = HuffmanEnc.huffman_table

        HammingEnc = HammingEncoder(EncodedData, 7, 4)
        HammingEncodedData = HammingEnc.encode()

        UselessEndBits = HammingEnc.useless_encoded_bits
        for i in range(USELESS_DATA_AMOUNT):
            HammingEncodedData.insert(0, (UselessEndBits >> (USELESS_DATA_AMOUNT - 1 - i)) & 1)

        Help.write_to_file(HammingEncodedData, DestinationFileName)

        with open(CodesTableFileName, mode="wb") as CodesTableFile:
                pickle.dump(CodesTable, CodesTableFile)

    @staticmethod
    def RunDecoder(InputFileName: str, DestinationFileName: str, CodesTableFileName: str):
        EncodedData = Help.read_from_file(InputFileName)

        with open(CodesTableFileName, "rb") as CodesTableFile:
            CodesTable = pickle.load(CodesTableFile)

        UselessEndBits = 0
        for i in range(USELESS_DATA_AMOUNT):
            UselessEndBits += EncodedData.pop(0) << i

        HammingDec = HammingDecoder(EncodedData, 7, 4)
        HammingDecodedData = HammingDec.decode()

        for i in range(UselessEndBits):
            HammingDecodedData.pop()

        HuffmanDec = HuffmanDecoder(HammingDecodedData, CodesTable)
        DecodedSequence = HuffmanDec.decode()

        with open(DestinationFileName, "w", encoding="utf-8") as DestinationFile:
            DestinationFile.write("".join(DecodedSequence))

    @staticmethod
    def CreateNoise(InputFileName: str, DestinationFileName: str, ErrorChance: float):
        import random

        Data = Help.read_from_file(InputFileName)

        print("Corrupted bits: ", end="")

        Errors = 0
        for i in range(8, len(Data), 7):
            e = random.random()
            if e <= ErrorChance:
                Data[i] = not Data[i]
                Errors += 1

                print(str(i), end=" ")

        Help.write_to_file(Data, DestinationFileName)

    def TestRun(self):
        self.RunEncoder('files\origin', 'files\encoded_data', 'files\codemap')
        self.CreateNoise('files\encoded_data', 'files\encoded_data_with_noise', 0.02)
        self.RunDecoder('files\encoded_data_with_noise', 'files\decoded_text', 'files\codemap')

    def TestRun2Error(self):
        pass

if __name__ == '__main__':
    Menu().TestRun()
    print(1)
