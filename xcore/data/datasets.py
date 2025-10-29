import numpy as np
import hydra.utils
import torch

from typing import Tuple
from typing import Dict, Union
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk
from datasets import Dataset as dt
import random
import xcore.common.util as util

NULL_ID_FOR_COREF = -1


class xcoreDataset(Dataset):
    def __init__(self, name: str, path: str, batch_size, processed_dataset_path, tokenizer, **kwargs):
        super().__init__()
        self.stage = name
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True, add_prefix_space=True)
        self.max_doc_len = kwargs.get("max_doc_len", None)
        self.cross = kwargs.get("type", None) == "cross"
        self.book = kwargs.get("type", None) == "book"
        type = kwargs.get("type", None)
        if type == "book":
            self.split_size = kwargs.get("split", None)
        special_tokens_dict = {"additional_special_tokens": ["[SPEAKER_START]", "[SPEAKER_END]"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        try:
            self.set = load_from_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path + "_" + type + "/")
        except:
            if self.book:
                self.set = dt.from_pandas(util.ontonotes_to_dataframe(path))
            if self.cross:
                self.set = dt.from_pandas(util.temp_dataframe(path))
            if self.stage == "train":
                # self.set = self.set.map(self.cut_document_to_length, batched=False)
                self.set = self.prepare_data(self.set)
            self.set = self.set.map(self.encode, batched=False)
            self.set = self.set.remove_columns(column_names=["speakers"])
            if self.stage != "test":
                self.set.save_to_disk(hydra.utils.get_original_cwd() + "/" + processed_dataset_path + "_" + type + "/")

    def prepare_data(self, set):
        return set.filter(lambda x: len(self.tokenizer(x["tokens"])["input_ids"]) <= self.max_doc_len)

    def cut_document_to_length(self, set_element):
        encoded_text = self.tokenizer(set_element["tokens"], add_special_tokens=True, is_split_into_words=True)
        if len(encoded_text["input_ids"]) <= self.max_doc_len + 1:
            result = set_element
        else:
            last_index_input_id_in_sentence = encoded_text.token_to_word(self.max_doc_len)
            eos_indices = [end for end in set_element["EOS_indices"] if end < last_index_input_id_in_sentence]
            last_sentence_end = eos_indices[-3]
            result = {
                "doc_key": set_element["doc_key"],
                "tokens": set_element["tokens"][:last_sentence_end],
                "speakers": set_element["speakers"][:last_sentence_end],
                "EOS_indices": eos_indices[:-3],
            }
            new_clusters = []
            for cluster in set_element["clusters"]:
                new_cluster = []
                for span in cluster:
                    if span[1] < last_sentence_end:
                        new_cluster.append(span)
                if len(new_cluster) >= 1:
                    new_clusters.append(new_cluster)
            result["clusters"] = new_clusters
        return result

    def _tokenize(self, tokens, clusters, speakers, eos_indices):
        token_to_new_token_map = []  # len() = len(tokens), contains indices of original sequence to new sequence
        new_token_map = []  # len() = len(new_tokens), contains indices of new sequence
        new_tokens = []  # contains new tokens
        last_speaker = None

        for idx, (token, speaker) in enumerate(zip(tokens, speakers)):
            if last_speaker != speaker:
                new_tokens += ["[SPEAKER_START]", speaker, "[SPEAKER_END]"]
                new_token_map += [None, None, None]
                last_speaker = speaker
            token_to_new_token_map.append(len(new_tokens))
            new_token_map.append(idx)
            new_tokens.append(token)

        for cluster in clusters:
            for start, end in cluster:
                assert tokens[start : end + 1] == new_tokens[token_to_new_token_map[start] : token_to_new_token_map[end] + 1]
        encoded_text = self.tokenizer(new_tokens, add_special_tokens=True, is_split_into_words=True)

        clusters = [
            [
                (
                    encoded_text.word_to_tokens(token_to_new_token_map[start]).start,
                    encoded_text.word_to_tokens(token_to_new_token_map[end]).end - 1,
                )
                for start, end in cluster
            ]
            for cluster in clusters
        ]
        eos_indices = [
            encoded_text.word_to_tokens(token_to_new_token_map[eos - 1]).start
            for eos in eos_indices
            if encoded_text.word_to_tokens(token_to_new_token_map[eos - 1]) != None
        ]
        output = {
            "tokens": tokens,
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "gold_clusters": clusters,
            "subtoken_map": encoded_text.word_ids(),
            "new_token_map": new_token_map,
            "EOS_indices": eos_indices,
        }
        return output

    def encode(self, example):
        if "clusters" not in example:
            example["clusters"] = []
        encoded = self._tokenize(
            example["tokens"],
            example["clusters"],
            example["speakers"],
            example["EOS_indices"],
        )

        if self.cross:
            encoded["sentence2doc"] = example["sentence2doc"]
        encoded["num_clusters"] = len(encoded["gold_clusters"]) if encoded["gold_clusters"] else 0
        encoded["max_cluster_size"] = max(len(c) for c in encoded["gold_clusters"]) if encoded["gold_clusters"] else 0
        encoded["length"] = len(encoded["input_ids"])
        return encoded

    def __len__(self) -> int:
        return self.set.shape[0]

    def __getitem__(self, index) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.set[index]

    # takes length of sequence (int) and eos_indices ([])
    # returns len x len zeros matrix with 1 in pos (start, all possible ends)
    def eos_mask(self, input_ids_len, eos_indices):
        mask = np.zeros((input_ids_len, input_ids_len))
        prec = 0
        for eos_idx in eos_indices:
            for i in range(prec, eos_idx + 1):
                for j in range(prec, eos_idx + 1):
                    if i != eos_indices[-1] and j != eos_indices[-1]:
                        mask[i][j] = 1
            prec = eos_idx
        mask = np.triu(mask)
        return mask

    # takes length of sequence (int) and coreferences ([[()]])
    # returns len x len zeros matrix with 1 in pos (start, end)
    def create_mention_matrix(self, input_ids_len, coreferences):
        matrix = np.zeros((input_ids_len, input_ids_len))
        for cluster in coreferences:
            for start_bpe_idx, end_bpe_idx in cluster:
                if start_bpe_idx < input_ids_len and end_bpe_idx < input_ids_len:
                    matrix[start_bpe_idx][end_bpe_idx] = 1
        return matrix

    # takes length of sequence (int) and coreferences ([[()]])
    # returns len zeros matrix with 1 in start position
    def create_start_matrix(self, input_ids_len, coreferences):
        matrix = np.zeros((input_ids_len))
        for cluster in coreferences:
            for start_bpe_idx, end_bpe_idx in cluster:
                matrix[start_bpe_idx] = 1
        return matrix

    # pad don't pad the rest, and is slow, think about something else
    def collate_fn(self, batch):
        batch = self.tokenizer.pad(batch)

        if self.cross == True:
            length = len(batch["input_ids"][0])
            batch["sentence2doc"] = sorted(
                list(
                    (int(sentence_idx), doc_id)
                    for sentence_idx, doc_id in batch["sentence2doc"][0].items()
                    if doc_id != None and int(sentence_idx) < len(batch["EOS_indices"][0])
                )
            )
            if len(batch["sentence2doc"]) == 0:
                slices_seq_index = []
            if len(batch["sentence2doc"]) == 1:
                slices_seq_index = [batch["EOS_indices"][0][batch["sentence2doc"][0][0]]]
            
            #     slices_seq_index = [
            #         batch["EOS_indices"][0][
            #             max(
            #                 [
            #                     t for (t, doc_id) in batch["sentence2doc"]
            #                     if doc_id == batch["sentence2doc"][0][1]
            #                 ]
            #             )
            #         ]
            #     ]
            # 
            # slices_seq_index.extend(
            #     [
            #         batch["EOS_indices"][0][sentence_idx] for (sentence_idx, doc_id) in batch["sentence2doc"]
            #         if doc_id != batch["sentence2doc"][sentence_idx - 1][1] 
            #         and sentence_idx != 0
            #         ]
            #     )
            
            
            slices_seq_index= [
                    batch["EOS_indices"][0][sentence_idx] for (sentence_idx, doc_id) in batch["sentence2doc"]
                    if sentence_idx < len(batch["EOS_indices"][0])-1 and doc_id != batch["sentence2doc"][sentence_idx+1][1] 
                    ]
            
            # batch["sentence2doc"] = sorted(list((int(t), v) for t, v in batch["sentence2doc"][0].items() if v != None))
            # temp = batch["EOS_indices"][0]
            # t3 = batch["sentence2doc"]
            # slices_seq_index = [temp[max([t for (t, v) in t3 if v == t3[0][1]])]]

            # slices_seq_index.extend([temp[t] for (t, v) in t3 if v != t3[t - 1][1] and t != 0])

            if len(slices_seq_index) == 0 or slices_seq_index[-1] != length:
                slices_seq_index.append(length)

        if self.book == True:
            length = len(batch["input_ids"][0])

            found = False
            while found != True:
                if self.stage == "train":
                    max_seq_len = random.randint(200, 1500)
                else:
                    max_seq_len = self.split_size
                    if self.split_size < 100:
                        max_seq_len = int(length / self.split_size) + 1

                if max_seq_len > length - 3:
                    max_seq_len = length + 1

                found = sum(
                    [
                        1 if len([item for item in batch["EOS_indices"][0] if item > step]) != 0 else 0
                        for step in range(max_seq_len, length, max_seq_len)
                    ]
                ) == len(
                    [
                        1 if len([item for item in batch["EOS_indices"][0] if item > step]) != 0 else 0
                        for step in range(max_seq_len, length, max_seq_len)
                    ]
                )

            slices_seq_index = [
                [item for item in batch["EOS_indices"][0] if item > step][0] for step in range(max_seq_len, length, max_seq_len)
            ]
            if len(slices_seq_index) == 0 or slices_seq_index[-1] != length:
                slices_seq_index.append(length)

        slices_seq_index = sorted(list(set(slices_seq_index)))
        prev = 0
        index_input_ids = []
        index_attention_mask = []
        index_eos_mask = []
        index_gold_mentions = []
        index_gold_starts = []
        index_padded_clusters = []
        temp = []
        tempppp = []
        gold_c = batch["gold_clusters"][0]
        i = 0
        index_tokens = []
        index_subtoken_map = []
        index_new_token_map = []
        for index in slices_seq_index:
            index_input_ids.append(batch["input_ids"][0][prev:index])
            index_attention_mask.append(batch["attention_mask"][0][prev:index])
            index_eos_mask.append(
                self.eos_mask(
                    len(index_input_ids[-1]),
                    [item - prev for item in batch["EOS_indices"][0] if item > prev and item <= index],
                )
            )
            off = batch["subtoken_map"][0][prev]
            if off == None:
                off = 0
            index_subtoken_map.append([i - off if i != None else None for i in batch["subtoken_map"][0][prev:index]])
            if prev == 0 and index == len(batch["subtoken_map"][0]):
                index_tokens.append(batch["tokens"][0])
                index_new_token_map.append(batch["new_token_map"][0])
            elif prev == 0:
                index_tokens.append(batch["tokens"][0][: batch["new_token_map"][0][batch["subtoken_map"][0][index]]])
                index_new_token_map.append(batch["new_token_map"][0][: batch["subtoken_map"][0][index]])
            elif index == len(batch["subtoken_map"][0]):
                index_tokens.append(batch["tokens"][0][batch["new_token_map"][0][batch["subtoken_map"][0][prev]] :])
                index_new_token_map.append(
                    [
                        i - batch["new_token_map"][0][batch["subtoken_map"][0][prev]]
                        for i in batch["new_token_map"][0][batch["subtoken_map"][0][prev] :]
                    ]
                )
            else:
                index_tokens.append(
                    batch["tokens"][0][
                        batch["new_token_map"][0][batch["subtoken_map"][0][prev]] : batch["new_token_map"][0][
                            batch["subtoken_map"][0][index]
                        ]
                    ]
                )
                index_new_token_map.append(
                    [
                        c - batch["new_token_map"][0][batch["subtoken_map"][0][prev]]
                        for c in batch["new_token_map"][0][batch["subtoken_map"][0][prev] : batch["subtoken_map"][0][index]]
                    ]
                )
            index_gold_clusters = [
                [
                    [item[0] - prev, item[1] - prev]
                    for item in cluster
                    if (item[0] - prev) >= 0
                    and (item[0] - prev) < (index - prev)
                    and (item[1] - prev) >= 0
                    and (item[1] - prev) < (index - prev)
                ]
                for cluster in batch["gold_clusters"][0]
            ]
            gold_c = [
                [
                    (
                        [prev, item[0] - prev, item[1] - prev]
                        if (item[0] - prev) >= 0
                        and (item[0] - prev) < (index - prev)
                        and (item[1] - prev) >= 0
                        and (item[1] - prev) < (index - prev)
                        else item
                    )
                    for item in cluster
                ]
                for cluster in gold_c
            ]
            index_gold_clusters = [elem for elem in index_gold_clusters if len(elem) > 0]

            index_gold_mentions.append(self.create_mention_matrix(len(index_input_ids[-1]), index_gold_clusters))
            index_gold_starts.append(self.create_start_matrix(len(index_input_ids[-1]), index_gold_clusters))

            index_num_clusters = len(index_gold_clusters)
            index_max_cluster_size = max(len(c) for c in index_gold_clusters) if len(index_gold_clusters) > 0 else 0
            tempppp.append(index_gold_clusters)
            if index_num_clusters == 0:
                index_padded_clusters.append([])
            else:
                index_padded_clusters.append(
                    [pad_clusters(cluster, index_num_clusters, index_max_cluster_size) for cluster in [index_gold_clusters]]
                )

            prev = index
            i += 1
            temp.append(index)

        max_num_clusters, max_max_cluster_size = max(batch["num_clusters"]), max(batch["max_cluster_size"])
        if max_num_clusters == 0:
            padded_clusters = []
        else:
            padded_clusters = [
                pad_clusters(cluster, max_num_clusters, max_max_cluster_size) for cluster in batch["gold_clusters"]
            ]

        gold_c = [[item for item in cluster if len(item) == 3] for cluster in gold_c]
        # assert [[[item[0] + item[1], item[0] + item[2]] for item in cluster] for cluster in gold_c] == batch["gold_clusters"][0]
        if max_num_clusters == 0:
            gold_c = []
        else:
            gold_c = [pad_clusters_v2(cluster, max_num_clusters, max_max_cluster_size) for cluster in [gold_c]]
        output = {
            "input_ids": torch.tensor(batch["input_ids"]),
            "index_input_ids": [torch.tensor(item) for item in index_input_ids],
            "attention_mask": torch.tensor(batch["attention_mask"]),
            "t_tokens": index_tokens,
            "t_subtoken_map": index_subtoken_map,
            "t_new_token_map": index_new_token_map,
            "index_attention_mask": [torch.tensor(item) for item in index_attention_mask],
            "index_eos_mask": [torch.tensor(item) for item in index_eos_mask],
            "index_gold_mentions": [torch.tensor(item) for item in index_gold_mentions],
            "index_gold_starts": [torch.tensor(item) for item in index_gold_starts],
            "gold_clusters": torch.tensor(padded_clusters),
            "gold_c": torch.tensor(gold_c),
            "index_gold_clusters": [torch.tensor(item) for item in index_padded_clusters],
            "temp": [0] + temp,
        }

        if self.stage == "train" and "bookcoref" in self.path:
            output["gold_mentions"] = torch.tensor(
                self.create_mention_matrix(len(batch["input_ids"][0]), batch["gold_clusters"][0])
            ).unsqueeze(0)
            output["gold_starts"] = torch.tensor(
                self.create_start_matrix(len(batch["input_ids"][0]), batch["gold_clusters"][0])
            ).unsqueeze(0)

        output["tempppp"] = tempppp
        output["tokens"] = batch["tokens"]
        output["doc_key"] = batch["doc_key"]
        output["subtoken_map"] = batch["subtoken_map"]
        output["new_token_map"] = batch["new_token_map"]
        output["eos_indices"] = torch.tensor(batch["EOS_indices"])
        output["singletons"] = "litbank" in self.path or "scico" in self.path
        output["added"] = batch["gold_clusters"]
        return output



def pad_clusters_inside(clusters, max_cluster_size):
    return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster)) for cluster in clusters]


def pad_clusters_inside_v2(clusters, max_cluster_size):
    return [
        cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster))
        for cluster in clusters
    ]

def pad_clusters_outside(clusters, max_num_clusters):
    return clusters + [[]] * (max_num_clusters - len(clusters))


def pad_clusters(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside(clusters, max_cluster_size)
    return clusters

def pad_clusters_v2(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside_v2(clusters, max_cluster_size)
    return clusters


