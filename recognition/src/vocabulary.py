
class Vocabulary:
    """Class responsible for encoding strings into integers and decoding them back"""

    def __init__(self, separate_characters, aux_tokens_count=0):
        """Gives aux_tokens_count elements in the front for encoding"""
        self.char_to_label = dict()
        self.label_to_char = dict()
        self.vocabulary_size = len(separate_characters)
        self.aux_tokens_count = aux_tokens_count

        for i, symbol in enumerate(separate_characters):
            self.char_to_label[symbol] = i + aux_tokens_count
            self.label_to_char[i + aux_tokens_count] = symbol

    def encode_string(self, string):
        result = []
        for symbol in string:
            if symbol in self.char_to_label:
                result.append(self.char_to_label[symbol])
            else:
                raise ValueError("Character {} not from vocabulary".format(symbol))
        return result

    def decode_string(self, seq):
        result = ""
        for element in seq:
            if element >= self.aux_tokens_count:
                result += self.label_to_char[element]
        return result
