import functools
from dataclasses import dataclass
from typing import Dict, Set


@dataclass(frozen=True)
class MochaTimitInventory:
    phones: Set[str]
    phonemes: list[str]  # see `clean_phon_feats.py`
    mapping: Dict[str, tuple[str, ...]]  # phone -> phoneme(s)


@functools.cache
def uk_english_inventory(broad_trans=True):
    _phones = {
        'm', 'mʲ', 'm̩', 'ɱ', 'n', 'n̩', 'ɲ', 'ŋ', 'p', 'pʲ', 'pʰ', 'pʷ', 'b', 'bʲ',
        't', 't̪', 'tʰ', 'tʲ', 'tʷ', 'd', 'd̪', 'dʲ', 'c', 'cʰ', 'cʷ', 'ɟ', 'ɟʷ', 'k',
        'kʰ', 'kʷ', 'ɡ', 'ɡʷ', 'ʔ', 'tʃ', 'dʒ', 's', 'z', 'ʃ', 'ʒ', 'f', 'fʲ', 'fʷ',
        'v', 'vʲ', 'vʷ', 'θ', 'ð', 'ç', 'h', 'w', 'ɹ', 'j', 'l', 'ɫ', 'ɫ̩', 'ʎ',

        'i', 'ʉ', 'ɪ', 'ʊ', 'e', 'ə', 'ɛ', 'ɜ', 'ɐ', 'a', 'ɑ', 'ɒ',
        'iː', 'ʉː', 'ɛː', 'ɜː', 'ɑː', 'ɒː',

        'aj', 'aw', 'ej', 'ɔj', 'əw',

        '', 'spn'
    }

    _phonemes = [
        'm', 'n', 'ŋ', 'p', 'b', 't', 'd', 'k', 'ɡ', 't͡ʃ', 'd͡ʒ', 's', 'z', 'ʃ', 'ʒ',
        'f', 'v', 'θ', 'ð', 'h', 'w', 'ɹ', 'j', 'l',

        'i', 'u', 'ɪ', 'ʊ', 'e', 'ə', 'ɛ', 'ɚ', 'ʌ', 'æ', 'ɑ', 'a', 'ɔ', 'ɑː', 'iː',
        'ɛː', 'uː', 'ɔː',

        'o', 'ʍ', 'ɾ',

        'ʔ',

        'sil', 'spn'
    ]

    _mapping = {
        'm': ('m',),
        'mʲ': ('m',),
        'm̩': ('ə', 'm') if broad_trans else ('m',),
        'ɱ': ('m',),
        'n': ('n',),
        'n̩': ('ə', 'n') if broad_trans else ('n',),
        'ɲ': ('n', 'j'),
        'ŋ': ('ŋ',),
        'p': ('p',),
        'pʲ': ('p',),
        'pʰ': ('p',),
        'pʷ': ('p',),
        'b': ('b',),
        'bʲ': ('b',),
        't': ('t',),
        't̪': ('t',),
        'tʰ': ('t',),
        'tʲ': ('t',),
        'tʷ': ('t',),
        'd': ('d',),
        'd̪': ('d',),
        'dʲ': ('d',),
        'c': ('k',),
        'cʰ': ('k',),
        'cʷ': ('k',),
        'k': ('k',),
        'kʰ': ('k',),
        'kʷ': ('k',),
        'ç': ('k',),
        'ɟ': ('ɡ',),
        'ɟʷ': ('ɡ',),
        'ɡ': ('ɡ',),
        'ɡʷ': ('ɡ',),

        # keep the glottal stop as a phoneme: sometimes it is an allophone of /t/
        #  (t-glottalisation), but it can also be an open juncture.
        'ʔ': ('ʔ',),

        'tʃ': ('t͡ʃ',),
        'dʒ': ('d͡ʒ',),
        's': ('s',),
        'z': ('z',),
        'ʃ': ('ʃ',),
        'ʒ': ('ʒ',),
        'f': ('f',),
        'fʲ': ('f',),
        'fʷ': ('f',),
        'v': ('v',),
        'vʲ': ('v',),
        'vʷ': ('v',),
        'θ': ('θ',),
        'ð': ('ð',),
        'h': ('h',),
        'w': ('w',),
        'ɹ': ('ɹ',),
        'j': ('j',),
        'l': ('l',),
        'ɫ': ('l',),
        'ɫ̩': ('ə', 'l') if broad_trans else ('l',),
        'ʎ': ('l', 'j'),

        'i': ('i',),
        'ʉ': ('u',),
        'ɪ': ('ɪ',),
        'ʊ': ('ʊ',),
        'e': ('e',),
        'ə': ('ə',),
        'ɛ': ('ɛ',),
        'ɜ': ('ɚ',),  # 'ɜ'
        'ɐ': ('ʌ',),
        'a': ('æ',),
        'ɑ': ('ɑ',),
        'ɒ': ('ɔ',),

        'iː': ('iː',),  # absent in the phoneme list (with phonological features)
        'ʉː': ('uː',),  # absent in the phoneme list (with phonological features)
        'ɛː': ('ɛː',),  # absent in the phoneme list (with phonological features)
        'ɜː': ('ɚ',),
        'ɑː': ('ɑː',),  # absent in the phoneme list (with phonological features)
        'ɒː': ('ɔː',),  # absent in the phoneme list (with phonological features)

        'aj': ('a', 'ɪ'),  # absent in the phoneme list (with phonological features)
        'aw': ('a', 'ʊ'),  # absent in the phoneme list (with phonological features)
        'ej': ('e', 'ɪ'),  # absent in the phoneme list (with phonological features)
        'ɔj': ('ɔ', 'ɪ'),  # absent in the phoneme list (with phonological features)
        'əw': ('ə', 'ʊ'),  # absent in the phoneme list (with phonological features)

        '': ('sil',),  # idem
        'spn': ('spn',),  # idem
    }

    return MochaTimitInventory(
        phones=_phones, phonemes=_phonemes, mapping=_mapping
    )


@functools.cache
def diph_english_inventory():
    # TODO: recompute this
    _phones = {
        'm', 'mʲ', 'm̩', 'ɱ', 'n', 'n̩', 'ɲ', 'ŋ', 'p', 'pʲ', 'pʰ', 'pʷ', 'b', 'bʲ',
        't', 't̪', 'tʰ', 'tʲ', 'tʷ', 'd', 'd̪', 'dʲ', 'c', 'cʰ', 'cʷ', 'ɟ', 'ɟʷ', 'k',
        'kʰ', 'kʷ', 'ɡ', 'ɡʷ', 'ʔ', 'tʃ', 'dʒ', 's', 'z', 'ʃ', 'ʒ', 'f', 'fʲ', 'fʷ',
        'v', 'vʲ', 'vʷ', 'θ', 'ð', 'ç', 'h', 'w', 'ɹ', 'j', 'l', 'ɫ', 'ɫ̩', 'ʎ',

        'i', 'ʉ', 'ɪ', 'ʊ', 'e', 'ə', 'ɛ', 'ɜ', 'ɐ', 'a', 'ɑ', 'ɒ',
        'iː', 'ʉː', 'ɛː', 'ɜː', 'ɑː', 'ɒː',

        'aj', 'aw', 'ej', 'ɔj', 'əw',

        '', 'spn'
    }

    # TODO: recompute this
    _phonemes = [
        'm', 'n', 'ŋ', 'p', 'b', 't', 'd', 'k', 'g', 't͡ʃ', 'd͡ʒ', 's', 'z', 'ʃ', 'ʒ',
        'f', 'v', 'θ', 'ð', 'h', 'w', 'ɹ', 'j', 'l',

        'i', 'u', 'ɪ', 'ʊ', 'e', 'ə', 'ɛ', 'ɚ', 'ʌ', 'æ', 'ɑ', 'ɔ',

        'ʍ', 'o',

        'sil', 'spn'
    ]

    _mapping = {
        'm': ('m',),
        'mʲ': ('m',),
        'm̩': ('m̩',),
        'ɱ': ('m',),
        'n': ('n',),
        'n̩': ('n̩',),
        'ɲ': ('n', 'j'),
        'ŋ': ('ŋ',),
        'p': ('p',),
        'pʲ': ('p',),
        'pʰ': ('p',),
        'pʷ': ('p',),
        'b': ('b',),
        'bʲ': ('b',),
        't': ('t',),
        't̪': ('t',),
        'tʰ': ('t',),
        'tʲ': ('t',),
        'tʷ': ('t',),
        'd': ('d',),
        'd̪': ('d',),
        'dʲ': ('d',),
        'c': ('k',),
        'cʰ': ('k',),
        'cʷ': ('k',),
        'k': ('k',),
        'kʰ': ('k',),
        'kʷ': ('k',),
        'ç': ('k',),
        'ɟ': ('ɡ',),
        'ɟʷ': ('ɡ',),
        'ɡ': ('ɡ',),
        'ɡʷ': ('ɡ',),

        # TODO: which phoneme for the glottal stop?
        'ʔ': ('ʔ',),

        'tʃ': ('t͡ʃ',),
        'dʒ': ('d͡ʒ',),
        's': ('s',),
        'z': ('z',),
        'ʃ': ('ʃ',),
        'ʒ': ('ʒ',),
        'f': ('f',),
        'fʲ': ('f',),
        'fʷ': ('f',),
        'v': ('v',),
        'vʲ': ('v',),
        'vʷ': ('v',),
        'θ': ('θ',),
        'ð': ('ð',),
        'h': ('h',),
        'w': ('w',),
        'ɹ': ('ɹ',),
        'j': ('j',),
        'l': ('l',),
        'ɫ': ('l',),
        'ɫ̩': ('l̩',),
        'ʎ': ('l', 'j'),

        'i': ('i',),
        'ʉ': ('u',),
        'ɪ': ('ɪ',),
        'ʊ': ('ʊ',),
        'e': ('ɛ',),
        'ə': ('ə',),
        'ɛ': ('ɛ',),
        'ɜ': ('ɚ',),
        'ɐ': ('ʌ',),
        'a': ('æ',),
        'ɑ': ('ɑ',),
        'ɒ': ('ɔ',),

        'iː': ('i',),  # absent in the phoneme list (with phonological features)
        'ʉː': ('u',),  # absent in the phoneme list (with phonological features)
        'ɛː': ('ɛ',),  # absent in the phoneme list (with phonological features)
        'ɜː': ('ɚ',),
        'ɑː': ('ɑ',),  # absent in the phoneme list (with phonological features)
        'ɒː': ('ɔ',),  # absent in the phoneme list (with phonological features)

        # absent in the phoneme list (with phonological features)
        'aj': ('aɪ1', 'aɪ2'),
        'aw': ('aʊ1', 'aʊ2'),
        'ej': ('eɪ1', 'eɪ2'),
        'ɔj': ('ɔɪ1', 'ɔɪ2'),
        'əw': ('əʊ1', 'əʊ2'),

        '': ('sil',),  # idem
        'spn': ('spn',),  # idem
    }

    return MochaTimitInventory(
        phones=_phones, phonemes=_phonemes, mapping=_mapping
    )


@functools.cache
def french_inventory():
    _phones = {
        'j', 'g', 'w', 's^', 'x', '_', 'u', 'i', 'o~', 'n', 'e^', 'e', 's', 'z^',
        't', 'q', 'v', 'h', 'k', 'p', 'b', 'a~', 'd', 'f', 'y', 'l', 'o^', 'e~',
        'x^', 'a', '__', 'r', 'x~', 'o', 'm', 'z'
    }

    _phonemes = [
        'y', 'ɛ', 'ɛ̃', 'a', 'œ', 'ø', 'u', 'ɑ̃', 'o', 'ə', 'ɔ', 'e', 'œ̃',
        'i', 'ɔ̃', 'ɥ', 'k', 't', 'p', 'w', 'j', 'f', 's', 'ʃ', 'b', 'm', 'n',
        'v', 'z', 'l', 'g', 'd', 'ʁ', 'ʒ', 'sil'
    ]

    _mapping = {
        'y': 'y',
        'e^': 'ɛ',
        'e~': 'ɛ̃',
        'a': 'a',
        'x^': 'œ',
        'x': 'ø',
        'u': 'u',
        'a~': 'ɑ̃',
        'o': 'o',
        'q': 'ə',
        'o^': 'ɔ',
        'e': 'e',
        'x~': 'œ̃',
        'i': 'i',
        'o~': 'ɔ̃',
        'h': 'ɥ',
        'k': 'k',
        't': 't',
        'p': 'p',
        'w': 'w',
        'j': 'j',
        'f': 'f',
        's': 's',
        's^': 'ʃ',
        'b': 'b',
        'm': 'm',
        'n': 'n',
        'v': 'v',
        'z': 'z',
        'l': 'l',
        'g': 'g',
        'd': 'd',
        'r': 'ʁ',
        'z^': 'ʒ',
        '_': 'sil',
        '__': 'sil',
    }

    return MochaTimitInventory(
        phones=_phones, phonemes=_phonemes, mapping=_mapping
    )
