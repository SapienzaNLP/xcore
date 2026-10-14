from typing import List, Optional

from transformers.tokenization_utils_base import TokenSpan


class WordEncoding:
    """Subword encoding of a pre-split word sequence.

    Mirrors the part of transformers' BatchEncoding that xCoRe uses: item access to
    input_ids / attention_mask, word_ids(), word_to_tokens() and token_to_word().
    """

    def __init__(self, input_ids: List[int], attention_mask: List[int], word_ids: List[Optional[int]]):
        self._data = {"input_ids": input_ids, "attention_mask": attention_mask}
        self._word_ids = word_ids
        self._word_spans = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            start = self._word_spans[word_idx].start if word_idx in self._word_spans else token_idx
            self._word_spans[word_idx] = TokenSpan(start, token_idx + 1)

    def __getitem__(self, key):
        return self._data[key]

    def word_ids(self) -> List[Optional[int]]:
        return list(self._word_ids)

    # None when the word produced no subtokens, matching BatchEncoding.word_to_tokens
    def word_to_tokens(self, word_index: int) -> Optional[TokenSpan]:
        return self._word_spans.get(word_index)

    def token_to_word(self, token_index: int) -> Optional[int]:
        return self._word_ids[token_index]


def wraps_with_cls_sep(tokenizer) -> bool:
    return tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None


def encode_words(tokenizer, words: List[str]) -> WordEncoding:
    """Encode words as [CLS] ... [SEP], the BERT/RoBERTa/DeBERTa layout, independent of the
    transformers version. Tokenizers without a CLS or SEP token get the bare subword ids.
    """
    # The checkpoints were trained on [CLS] ... [SEP] inputs. transformers 5.0-5.3 stopped
    # inserting them for DeBERTa-v3 with add_special_tokens=True, so we add them ourselves
    # to get the same layout from every transformers version.
    encoded = tokenizer(words, add_special_tokens=False, is_split_into_words=True)
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    word_ids = list(encoded.word_ids())
    if wraps_with_cls_sep(tokenizer):
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        attention_mask = [1] + attention_mask + [1]
        word_ids = [None] + word_ids + [None]
    return WordEncoding(input_ids, attention_mask, word_ids)
