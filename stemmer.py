import warnings
import re
class StaticArrays:
    rule_words = [
        "ই",
        "ও",
        "তো",
        "কে",
        "তে",
        "র",
        "রা",
        "য়",
        "দের",
        "ে.ে",
        "ের",
        "ে",
        "টি",
        "টির",
        "েরটা",
        "েরটার",
        "টা",
        "টার",
        "গুলো",
        "গুলোর",
        "েরগুলো",
        "েরগুলোর",
    ]
    rule_dict = {
        "রছ": "র",
        "রব": "র",
        "েয়ে": "া",
        "েয়েছিল": "া",
        "েয়েছিলেন": "া",
        "ে.েছিলেন": "া.",
        "ে.ে": "া.",
    }
class BanglaStemmer:
    def __init__(self):
        pass
    def _repetition_checker(self, word, lin):
        return word == lin
    def _len_checker(self, temp_arr):
        index = None
        word = None
        current_len = -1
        for i in range(0, len(temp_arr)):
            if len(temp_arr[i]) > current_len:
                current_len = len(temp_arr[i])
                index = i
                word = temp_arr[index]
        return word
    def stem(self, lin=""):
        if not isinstance(lin, str):
            warnings.warn(
                "stem() expected arg as a string, but got a non-string value."
            )
            return ""
        temp_arr = []
        for word in StaticArrays.rule_words:
            if re.search(".*" + word + "$", lin):
                temp_arr.append(word)
        if len(temp_arr) != 0:
            longest_word = self._len_checker(temp_arr)
            if StaticArrays.rule_dict.get(longest_word):
                sliced = lin.replace(longest_word, StaticArrays.rule_dict[longest_word])
                if self._repetition_checker(sliced, lin):
                    return lin[0] + "া" + lin[2] + "া"
                else:
                    return sliced
            else:
                new_index = len(lin) - len(longest_word)
                return lin[0:new_index]
        else:
            return lin
