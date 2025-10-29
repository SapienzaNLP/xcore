import torch
import numpy as np

from transformers import AutoTokenizer

from xcore.models.pl_modules import CrossPLModule
from xcore.common.util import *
from xcore.common.constants import *
from xcore.models import *

from transformers.utils.hub import cached_file as hf_cached_file

class xCoRe:
    def __init__(self, hf_name_or_path="sapienzanlp/xcore-litbank", device="cuda"):
        self.device = device
        path = self.__get_model_path__(hf_name_or_path)
        self.model = CrossPLModule.load_from_checkpoint(path, _recursive_=False, map_location=self.device)
        # self.model = CrossPLModule.load_from_checkpoint(hf_name_or_path, _recursive_=False, map_location=device)
        self.model = self.model.eval()
        self.model = self.model.model
        self.tokenizer = self.__get_model_tokenizer__()

    def __get_model_path__(self, hf_name_or_path):
        try:
            print(hf_name_or_path, "loading")
            path = hf_cached_file(hf_name_or_path, "weights.ckpt")
        except:
            print(hf_name_or_path, "not found on huggingface, loading from local path ")
            path = hf_name_or_path
        return path

    def __get_model_tokenizer__(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model.encoder_hf_model_name, use_fast=True, add_prefix_space=True)
        special_tokens_dict = {"additional_special_tokens": ["[SPEAKER_START]", "[SPEAKER_END]"]}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    def __sample_type__(self, sample):
        if isinstance(sample, str):
            result = "text"
        if isinstance(sample, list):
            result = "word_tokenized"
            if len(sample) != 0 and isinstance(sample[0], list):
                result = "sentence_tokenized"
        return result

    def preprocess(self, sample, mode):
        doc_len = []
        if mode == "short" or mode =="long":
            tokens, eos = self.preprocess_text(sample)
        elif mode == "cross":
            tokens = []
            eos = []
            offset = 0
            for elem in sample:
                elem_tok, elem_eos = self.preprocess_text(elem)
                tokens.extend(elem_tok)
                doc_len.extend([offset + len(elem_tok)])
                eos.extend([el + offset for el in elem_eos]) 
                offset += len(elem_tok)
        return tokens, eos, doc_len


    def preprocess_text(self, sample):
        text_type = self.__sample_type__(sample)
        char_offsets = None
        if text_type == "text":
            nlp = download_load_spacy()
            char_offsets = []
            sentences = []
            off = 0
            s = sent_tokenize(sample)
            for sent, sentence in zip(nlp.pipe(s), s):
                char_offsets.append([(off + tok.idx, off + tok.idx + len(tok.text) - 1) for tok in sent])
                sentences.append([tok.text for tok in sent])
                off += len(sentence) + 1
            char_offsets = flatten(char_offsets)
            tokens = flatten(sentences)
            eos_len = [len(value) for value in sentences]
            eos = [sum(eos_len[0 : (i[0] + 1)]) for i in enumerate(eos_len)]
        elif text_type == "word_tokenized":
            nlp = download_load_spacy()
            tokens = sample
            eos = [idx + 1 for idx, tok in enumerate(tokens) if tok == "."]
            if len(eos) == 0 or eos[-1] != len(tokens):
                eos.append(len(tokens))
        elif text_type == "sentence_tokenized":
            sentences = sample
            tokens = flatten(sentences)
            eos_len = [len(value) for value in sentences]
            eos = [sum(eos_len[0 : (i[0] + 1)]) for i in enumerate(eos_len)]
        return tokens, eos

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


    # enable speakers usage (only when we have tokenized input)
    @torch.no_grad()
    # def predict(self, sample, mode = "short", singletons=False, add_gold_clusters=None, predefined_mentions=None, speakers=None):
    def predict(self, sample, mode = "short", max_length = 4000, singletons=False):
        tokens, eos_indices, doc_len = self.preprocess(sample, mode)  # [[w1,w2,w3...], []]
        tokenized = self.tokenize(tokens, eos_indices, mode, doc_len, max_length)

        output = self.model(
            stage="test",
            input_ids=tokenized["index_input_ids"],
            attention_mask=tokenized["index_attention_mask"],
            eos_mask=tokenized["index_eos_mask"],
            singletons=singletons,
            temp=tokenized["temp"],
            tokens=tokenized["t_tokens"],
            subtoken_map=tokenized["t_subtoken_map"],
            new_token_map=tokenized["t_new_token_map"],
        )

        result = {}
        if mode != "cross":
            result["tokens"] = tokens
            clusters_predicted = original_token_offsets3(
                clusters=output["pred_dict"]["full_coreferences"],
                subtoken_map=tokenized["subtoken_map"][0],
                new_token_map=tokenized["new_token_map"][0],
            )

            result["clusters_token_offsets"] = clusters_predicted
            result["clusters_token_text"] = [
                [" ".join(tokens[span[0] : span[1] + 1]) for span in cluster] for cluster in clusters_predicted
            ]

        else:
            offset = 0
            result["tokens"] =[]
            for length in doc_len:
                result["tokens"].append([tokens[offset:length]]) 
                offset = length
            clusters_predicted = original_token_offsetst(
                mode,
                tokenized["temp"],
                clusters=output["pred_dict"]["full_coreferences"],
                subtoken_maps=tokenized["t_subtoken_map"],
                new_token_maps=tokenized["t_new_token_map"],
            )

            result["clusters_token_offsets"] = clusters_predicted

            clusters_predicted = original_token_offsets3(
                    clusters=output["pred_dict"]["full_coreferences"],
                    subtoken_map=tokenized["subtoken_map"][0],
                    new_token_map=tokenized["new_token_map"][0],
                )
            result["clusters_token_text"] = [
                [" ".join(tokens[span[0] : span[1] + 1]) for span in cluster] for cluster in clusters_predicted
            ]
        
        
        return result

    def create_mention_matrix(self, input_ids_len, mentions):
        matrix = np.zeros((input_ids_len, input_ids_len))
        for start_bpe_idx, end_bpe_idx in mentions:
            matrix[start_bpe_idx][end_bpe_idx] = 1
        return matrix

    def tokenize(self, tokens, eos_indices, mode, doc_len, max_seq_len = 4000):
        token_to_new_token_map = []  # len() = len(tokens), contains indices of original sequence to new sequence
        new_token_map = []  # len() = len(new_tokens), contains indices of new sequence
        new_tokens = []  # contains new tokens
        last_speaker = None

        speakers = ["-"] * len(tokens)
        for idx, (token, speaker) in enumerate(zip(tokens, speakers)):
            if last_speaker != speaker:
                new_tokens += ["[SPEAKER_START]", speaker, "[SPEAKER_END]"]
                new_token_map += [None, None, None]
                last_speaker = speaker
            token_to_new_token_map.append(len(new_tokens))
            new_token_map.append(idx)
            new_tokens.append(token)

        encoded_text = self.tokenizer(new_tokens, add_special_tokens=True, is_split_into_words=True)
        

        eos_indices = [
            encoded_text.word_to_tokens(token_to_new_token_map[eos - 1]).start
            for eos in eos_indices
            if encoded_text.word_to_tokens(token_to_new_token_map[eos - 1]) != None
        ]

        # test with litbank and ecb
        length = len(encoded_text["input_ids"])
        if mode == "cross":
            seq_index = []
            for l in doc_len:
                if l == doc_len[-1]:
                    seq_index.append(len(encoded_text["input_ids"]))
                else:
                    max_eos = [eos for eos in eos_indices if eos > l][0]
                    seq_index.append(max_eos)
            
        else:
            seq_index = []
            if max_seq_len > length - 3:
                max_seq_len = length + 1

            seq_index = [
                [item for item in eos_indices if item > step][0] for step in range(max_seq_len, length, max_seq_len)
            ]
            
        if len(seq_index) == 0 or seq_index[-1] != length:
                seq_index.append(length)
        
        subtoken_map= encoded_text.word_ids()
        slices_seq_index = sorted(list(set(seq_index)))
        prev = 0
        index_input_ids = []
        index_attention_mask = []
        index_eos_mask = []
        temp = []
        tempppp = []
        index_tokens = []
        index_subtoken_map = []
        index_new_token_map = []
        for index in slices_seq_index:
            index_input_ids.append(encoded_text["input_ids"][prev:index])
            index_attention_mask.append(encoded_text["attention_mask"][prev:index])
            index_eos_mask.append(
                self.eos_mask(
                    len(index_input_ids[-1]),
                    [item - prev for item in eos_indices if item > prev and item <= index],
                )
            )
            off = subtoken_map[prev]
            if off == None:
                off = 0
            index_subtoken_map.append([i - off if i != None else None for i in subtoken_map[prev:index]])
            if prev == 0 and index == len(subtoken_map):
                index_tokens.append(tokens)
                index_new_token_map.append(new_token_map)
            elif prev == 0:
                index_tokens.append(tokens[: new_token_map[subtoken_map[index]]])
                index_new_token_map.append(new_token_map[: subtoken_map[index]])
            elif index == len(subtoken_map):
                index_tokens.append(tokens[new_token_map[subtoken_map[prev]] :])
                index_new_token_map.append(
                    [
                        i - new_token_map[subtoken_map[prev]]
                        for i in new_token_map[subtoken_map[prev] :]
                    ]
                )
            else:
                index_tokens.append(
                    tokens[
                        new_token_map[subtoken_map[prev]] : new_token_map[
                            subtoken_map[index]
                        ]
                    ]
                )
                index_new_token_map.append(
                    [
                        c - new_token_map[subtoken_map[prev]]
                        for c in new_token_map[subtoken_map[prev] : subtoken_map[index]]
                    ]
                )
            
            prev = index
            temp.append(index)

        output = {
            "input_ids": encoded_text["input_ids"],
            "index_input_ids": [torch.tensor(item).to(self.device) for item in index_input_ids],
            "attention_mask": encoded_text["attention_mask"],
            "t_tokens": index_tokens,
            "t_subtoken_map": index_subtoken_map,
            "t_new_token_map": index_new_token_map,
            "index_attention_mask": [torch.tensor(item).to(self.device) for item in index_attention_mask],
            "index_eos_mask": [torch.tensor(item).to(self.device) for item in index_eos_mask],
            "temp": [0] + temp,
        }
        output["tempppp"] = tempppp
        output["tokens"] = tokens
        output["subtoken_map"] = encoded_text.word_ids(),
        output["new_token_map"] = new_token_map,
        output["eos_indices"] = self.eos_mask(len(encoded_text["input_ids"]), eos_indices),



        return output
