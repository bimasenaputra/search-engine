import array
from math import log

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_list = [postings_list[i] if i == 0 else postings_list[i] - postings_list[i-1] for i in range(len(postings_list))]
        encoded_postings = VBEPostings.vb_encode(gap_list)
        return bytearray(encoded_postings)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        numbers = []
        n =  0
        for i in range(len(encoded_bytestream)):
            if encoded_bytestream[i] < 128:
                n = (n << 7) + encoded_bytestream[i]
            else:
                n = (n << 7) + (encoded_bytestream[i] - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        gap_list_bytestream = VBEPostings.vb_decode(encoded_postings_list)
        encoded_id_list = [gap_list_bytestream[0]]
        for i in range(1, len(gap_list_bytestream)):
            encoded_id_list.append(encoded_id_list[i-1] + gap_list_bytestream[i])
        return encoded_id_list

class EliasGammaPostings:

    @staticmethod
    def gamma_encode_number(number):
        # Implementation: https://www.geeksforgeeks.org/elias-gamma-encoding-in-python/
        log2 = lambda x: log(x, 2)

        if number == 0: 
            return '0'
      
        n = 1 + int(log2(number))
        b = number - 2**(int(log2(number)))
      
        l = int(log2(number))
        
        s = '{0:0%db}' % l
        binary = s.format(b)
        unary = (n-1)*'0' +'1'
      
        binarystring = unary + binary

        # convert binary string to bytes 
        # https://stackoverflow.com/questions/32675679/convert-binary-string-to-bytearray-in-python-3
        return int(binarystring, 2).to_bytes((len(binarystring) + 7) // 8, byteorder='big')

    @staticmethod
    def gamma_encode(list_of_numbers):
        bytes = []
        for number in list_of_numbers:
            bytes.append(EliasGammaPostings.gamma_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        gap_list = [postings_list[i] if i == 0 else postings_list[i] - postings_list[i-1] for i in range(len(postings_list))]
        encoded_postings = EliasGammaPostings.gamma_encode(gap_list)
        return bytearray(encoded_postings)

    @staticmethod
    def gamma_decode(encoded_bytestream):
        # Implementation based on https://www.geeksforgeeks.org/elias-gamma-decoding-in-python/ with some modifications

        numbers = []
        i = 0
        while i < len(encoded_bytestream):
            while encoded_bytestream[i] == 0:
                i = i + 1

            j = i
            while j < len(encoded_bytestream) and encoded_bytestream[j] != 0:
                j = j + 1

            number = int.from_bytes(encoded_bytestream[i:j], byteorder='big')
            numbers.append(number)

            i = j
        
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        gap_list_bytestream = EliasGammaPostings.gamma_decode(encoded_postings_list)
        encoded_id_list = [gap_list_bytestream[0]]
        for i in range(1, len(gap_list_bytestream)):
            encoded_id_list.append(encoded_id_list[i-1] + gap_list_bytestream[i])
        return encoded_id_list

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        print("byte hasil encode: ", encoded_postings_list)
        print("ukuran encoded postings: ", len(encoded_postings_list), "bytes")
        decoded_posting_list = Postings.decode(encoded_postings_list)
        print("hasil decoding: ", decoded_posting_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        print()