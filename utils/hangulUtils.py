import os
import json
from pathlib import Path


static_path = Path(os.path.realpath(__file__)).parent.parent / 'static'


class HangulUtil:
    # Start: <start>, End: <end>
    with open(static_path / "jamo.json", "r") as j:
        word_map = json.loads(j.read())
        word_list = list(word_map)
        for c, i in word_map.items():
            word_list[i] = c

    @classmethod
    def vocabulary_size(cls):
        return len(cls.word_list)

    @classmethod
    def jamo_to_index(cls, letter):
        assert type(letter) is str

        if letter not in cls.word_map:
            raise ValueError("Letter not in the vocabulary: %s" % letter)

        return cls.word_map[letter]

    @classmethod
    def index_to_jamo(cls, index):
        assert type(index) is int

        return cls.word_list[index]

    @classmethod
    def decompose(cls, character):
        """ Decompose single hangul character to letters """
        assert type(character) is str and len(character) == 1
        # assert ord(u'가') <= chr(character) and chr(character) <= ord(u'힣')

        fir = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
        snd = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
        thr = [chr(0x3130)] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

        val = ord(character) - ord(u'가')

        res = []
        l1, l2, l3 = len(fir), len(snd), len(thr)

        res.append(thr[val%l3]); val = val // l3
        res.append(snd[val%l2]); val = val // l2
        res.append(fir[val])

        res = tuple(reversed(res))

        return res

    @classmethod
    def letter_to_caption(cls, character):
        """ Wrap the decomposed character with <start> and <end> """
        res = list()
        res.append(cls.jamo_to_index('<start>'))
        res += [cls.jamo_to_index(x) for x in cls.decompose(character)]
        res.append(cls.jamo_to_index('<end>'))
        return res


class KSXUtil:
    with open(static_path / "ksx1001.txt", "rt") as file:
        ksx_list = list(file.read().split('\n'))

    @classmethod
    def ksx_index(cls, letter):
        assert type(letter) is str and len(letter) == 1
        # assert ord(u'가') <= chr(letter) and chr(letter) <= ord(u'힣')

        idx = cls.ksx_list.index(letter)
        return idx
