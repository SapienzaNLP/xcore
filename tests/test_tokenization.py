import os

import pytest
import torch

from xcore.common.tokenization import TokenSpan, WordEncoding, encode_words

ENCODER = "microsoft/deberta-v3-large"
SPEAKER_TOKENS = ["[SPEAKER_START]", "[SPEAKER_END]"]
WORDS = ["[SPEAKER_START]", "-", "[SPEAKER_END]", "John", "met", "Mary", "in", "Zürich", ".", "He", "liked", "her", "."]

# Output of transformers 4.57.6 for WORDS with add_special_tokens=True, i.e. the layout
# the published checkpoints were trained on.
REFERENCE_INPUT_IDS = [1, 128001, 341, 128002, 918, 1626, 2612, 267, 71875, 323, 383, 3172, 342, 323, 2]
REFERENCE_WORD_IDS = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(ENCODER, use_fast=True, add_prefix_space=True)
    except OSError as e:
        pytest.skip(f"{ENCODER} tokenizer unavailable: {e}")
    tok.add_special_tokens({"additional_special_tokens": SPEAKER_TOKENS})
    return tok


def test_encode_words_matches_transformers4_layout(tokenizer):
    encoded = encode_words(tokenizer, WORDS)
    assert encoded["input_ids"] == REFERENCE_INPUT_IDS
    assert encoded["attention_mask"] == [1] * len(REFERENCE_INPUT_IDS)
    assert encoded.word_ids() == REFERENCE_WORD_IDS
    assert [encoded.word_to_tokens(i) for i in range(len(WORDS))] == [
        TokenSpan(i + 1, i + 2) for i in range(len(WORDS))
    ]


def test_encode_words_matches_add_special_tokens_when_tokenizer_inserts_them(tokenizer):
    words = ["Antidisestablishmentarianism", "puzzled", "Dr.", "O'Neill", "on", "2024-01-05", "."]
    reference = tokenizer(words, add_special_tokens=True, is_split_into_words=True)
    if reference["input_ids"][0] != tokenizer.cls_token_id:
        pytest.skip("this transformers version omits [CLS]/[SEP] with add_special_tokens=True")
    encoded = encode_words(tokenizer, words)
    assert encoded["input_ids"] == reference["input_ids"]
    assert encoded["attention_mask"] == reference["attention_mask"]
    assert encoded.word_ids() == reference.word_ids()
    for i in range(len(words)):
        assert tuple(encoded.word_to_tokens(i)) == tuple(reference.word_to_tokens(i))
    for t in range(len(reference["input_ids"])):
        assert encoded.token_to_word(t) == reference.token_to_word(t)


def test_multi_subtoken_word_span(tokenizer):
    encoded = encode_words(tokenizer, ["Antidisestablishmentarianism", "ends"])
    span = encoded.word_to_tokens(0)
    assert span.start == 1
    assert span.end - span.start > 1
    assert encoded.word_to_tokens(1) == TokenSpan(span.end, span.end + 1)
    assert encoded["input_ids"][0] == tokenizer.cls_token_id
    assert encoded["input_ids"][-1] == tokenizer.sep_token_id


class _NoSpecialTokensTokenizer:
    cls_token_id = None
    sep_token_id = None

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)


def test_encode_words_without_cls_sep_leaves_ids_unwrapped(tokenizer):
    encoded = encode_words(_NoSpecialTokensTokenizer(tokenizer), WORDS)
    assert encoded["input_ids"] == REFERENCE_INPUT_IDS[1:-1]
    assert encoded.word_ids() == REFERENCE_WORD_IDS[1:-1]
    assert encoded.word_to_tokens(0) == TokenSpan(0, 1)


def test_word_encoding_missing_word_returns_none():
    encoded = WordEncoding([1, 10, 11, 12, 2], [1] * 5, [None, 0, 0, 2, None])
    assert encoded.word_to_tokens(0) == TokenSpan(1, 3)
    assert encoded.word_to_tokens(1) is None
    assert encoded.word_to_tokens(2) == TokenSpan(3, 4)
    assert encoded.token_to_word(0) is None
    assert encoded.token_to_word(2) == 0


def test_dataset_tokenize_offsets_include_cls(tokenizer):
    from xcore.data.datasets import xcoreDataset

    dataset = xcoreDataset.__new__(xcoreDataset)
    dataset.tokenizer = tokenizer
    output = dataset._tokenize(
        tokens=["John", "met", "Mary", "."],
        clusters=[[(0, 0), (2, 2)]],
        speakers=["-"] * 4,
        eos_indices=[4],
    )
    # [CLS] [SPEAKER_START] - [SPEAKER_END] John met Mary . [SEP]
    assert output["input_ids"] == [1, 128001, 341, 128002, 918, 1626, 2612, 323, 2]
    assert output["gold_clusters"] == [[(4, 4), (6, 6)]]
    assert output["EOS_indices"] == [7]
    assert output["subtoken_map"] == [None, 0, 1, 2, 3, 4, 5, 6, None]


@pytest.mark.parametrize("first_id, stale", [(128001, True), (1, False)])
def test_dataset_rejects_cache_without_cls(monkeypatch, tokenizer, first_id, stale):
    from datasets import Dataset as dt
    from xcore.data import datasets as xcore_datasets

    monkeypatch.setattr(xcore_datasets.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(xcore_datasets.hydra.utils, "get_original_cwd", lambda: "/nonexistent")
    monkeypatch.setattr(xcore_datasets, "load_from_disk", lambda path: dt.from_dict({"input_ids": [[first_id, 918, 2]]}))

    def build():
        return xcore_datasets.xcoreDataset("train", "unused", 1, "data/cache/train", ENCODER, type="book")

    if stale:
        with pytest.raises(RuntimeError, match="without \\[CLS\\]/\\[SEP\\]"):
            build()
    else:
        assert build().set[0]["input_ids"][0] == 1


def test_book_dataset_build_reload_and_collate(monkeypatch, tmp_path, tokenizer):
    import json

    from xcore.data import datasets as xcore_datasets

    monkeypatch.setattr(xcore_datasets.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(xcore_datasets.hydra.utils, "get_original_cwd", lambda: str(tmp_path))
    doc = {
        "doc_key": "d1",
        "sentences": [["John", "met", "Mary", "."], ["He", "liked", "her", "."]],
        "clusters": [[[0, 0], [4, 4]], [[2, 2], [6, 6]]],
    }
    (tmp_path / "doc.jsonlines").write_text(json.dumps(doc) + "\n")

    def build():
        return xcore_datasets.xcoreDataset("val", "doc.jsonlines", 1, "cache/val", ENCODER, type="book", split=4000)

    build()
    assert (tmp_path / "cache" / "val_book").exists()
    dataset = build()  # loads the saved cache through the stale-cache guard
    # [CLS] [SPEAKER_START] - [SPEAKER_END] John met Mary . He liked her . [SEP]
    assert dataset[0]["input_ids"][0] == tokenizer.cls_token_id
    assert dataset[0]["gold_clusters"] == [[[4, 4], [8, 8]], [[6, 6], [10, 10]]]
    assert dataset[0]["EOS_indices"] == [7, 11]

    batch = dataset.collate_fn([dataset[0]])
    assert batch["input_ids"].shape == (1, 13)
    assert len(batch["index_input_ids"]) == 1
    mentions = batch["index_gold_mentions"][0]
    assert {tuple(idx) for idx in mentions.nonzero().tolist()} == {(4, 4), (8, 8), (6, 6), (10, 10)}
    assert batch["gold_clusters"].tolist() == [[[[4, 4], [8, 8]], [[6, 6], [10, 10]]]]


class _Stop(Exception):
    pass


def test_encoder_loaded_in_float32(monkeypatch):
    from xcore.models import model_cross

    captured = {}

    def fake_from_pretrained(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(model_cross.AutoModel, "from_pretrained", fake_from_pretrained)
    with pytest.raises(_Stop):
        model_cross.xCoRe_system(huggingface_model_name=ENCODER)
    assert captured["name"] == ENCODER
    assert captured["dtype"] == torch.float32


@pytest.mark.skipif(not os.environ.get("XCORE_SLOW_TESTS"), reason="set XCORE_SLOW_TESTS=1 to download and run the checkpoint")
def test_litbank_checkpoint_predicts_clusters():
    from xcore import xCoRe

    model = xCoRe("sapienzanlp/xcore-litbank", device="cpu")
    assert next(model.model.encoder.parameters()).dtype == torch.float32
    result = model.predict(
        "Mary met John at the station in the early morning. She told him that her train had been late again, "
        "and he laughed at her because he had warned her about it. The station master watched them. "
        "He had seen the couple there many times, and he knew that John always arrived first. "
        "Later that day, Mary wrote to her sister that John had been kind to her.",
        mode="long",
    )
    # Clusters produced by the pre-change code on transformers 4.57.6.
    predicted = sorted(sorted(tuple(span) for span in cluster) for cluster in result["clusters_token_offsets"])
    assert predicted == [
        [(2, 2), (13, 13), (23, 23), (28, 28), (54, 54), (69, 69)],
        [(4, 5), (46, 46)],
        [(11, 11), (15, 15), (26, 26), (31, 31), (63, 63), (66, 66), (74, 74)],
        [(35, 37), (41, 41), (51, 51)],
        [(39, 39), (44, 45)],
    ]
