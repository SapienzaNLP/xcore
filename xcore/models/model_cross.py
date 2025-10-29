import torch
import numpy as np
import random

from transformers import AutoModel, AutoConfig
import math
from xcore.common.util import *
from xcore.common.constants import *

from torch.nn import init
from torch import nn
from transformers import AutoModel, AutoConfig

from xcore.common.util import *
from xcore.common.constants import *

from transformers import (
                DistilBertModel,
                DistilBertConfig,
            )

class attention(torch.nn.Module):
    def __init__(self, model, representation):
        super().__init__()
        self.model = model
        self.t = RepresentationLayer(
            type="FC",  # fullyconnected
            input_dim=2048,
            output_dim=768,
            hidden_dim=1024,
        )

        self.representation = representation

    def forward(self, input):
        tt = self.t(input)
        if tt.shape[1] > 512:
            tt = tt[0][:511].unsqueeze(0)
        logits = self.model(inputs_embeds=tt).last_hidden_state[0][0].unsqueeze(0)
        return logits


class xCoRe_system(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # document transformer encoder
        self.encoder_hf_model_name = kwargs["huggingface_model_name"]
        self.encoder = AutoModel.from_pretrained(self.encoder_hf_model_name)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_hf_model_name)
        self.encoder.resize_token_embeddings(self.encoder.embeddings.word_embeddings.num_embeddings + 3)
        self.device = self.encoder.device

        # freeze
        if kwargs["freeze_encoder"]:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.negatives = kwargs.get("negatives", False)

        self.cluster_representation = "transformer"
        # span representation, now is concat_start_end
        self.span_representation = kwargs["span_representation"]
        # type of representation layer in 'Linear, FC, LSTM-left, LSTM-right, Conv1d'
        self.representation_layer_type = "FC"  # fullyconnected
        # span hidden dimension
        self.token_hidden_size = self.encoder_config.hidden_size

        # if span representation method is to concatenate start and end, a mention hidden size will be 2*token_hidden_size
        if self.span_representation == "concat_start_end":
            self.mention_hidden_size = self.token_hidden_size * 2

        self.mes = True

        self.num_cats = len(CATEGORIES) + 1  # +1 for ALL
        self.all_cats_size = self.token_hidden_size * self.num_cats

        self.antecedent_s2s_all_weights = nn.Parameter(
                torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
            )
        self.antecedent_e2e_all_weights = nn.Parameter(
                torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
            )
        self.antecedent_s2e_all_weights = nn.Parameter(
                torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
            )
        self.antecedent_e2s_all_weights = nn.Parameter(
                torch.empty((self.num_cats, self.token_hidden_size, self.token_hidden_size))
            )

        self.antecedent_s2s_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))
        self.antecedent_e2e_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))
        self.antecedent_s2e_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))
        self.antecedent_e2s_all_biases = nn.Parameter(torch.empty((self.num_cats, self.token_hidden_size)))

        self.coref_start_all_mlps = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=1,
            hidden_dim=self.mention_hidden_size,
        )

        self.coref_end_all_mlps = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=1,
            hidden_dim=self.mention_hidden_size,
        )

        # mention extraction layers
        # representation of start token
        self.start_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        # representation of end token
        self.end_token_representation = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        self.antecedent_s2s_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        self.antecedent_e2e_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        self.antecedent_s2e_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )
        self.antecedent_e2s_classifier = RepresentationLayer(
            type=self.representation_layer_type,  # fullyconnected
            input_dim=self.token_hidden_size,
            output_dim=self.token_hidden_size,
            hidden_dim=self.token_hidden_size,
        )

        # models probability to be the start of a mention
        self.start_token_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.token_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
        )

        # model mention probability from start and end representations
        self.start_end_classifier = RepresentationLayer(
            type=self.representation_layer_type,
            input_dim=self.mention_hidden_size,
            output_dim=1,
            hidden_dim=self.token_hidden_size,
        )

        
        self.t_coref_representation = RepresentationLayer(
                type=self.representation_layer_type,
                input_dim=self.mention_hidden_size,
                output_dim=self.mention_hidden_size,
                hidden_dim=self.mention_hidden_size,
            )

        self.tt_coref_representation = RepresentationLayer(
                type=self.representation_layer_type,
                input_dim=self.mention_hidden_size,
                output_dim=self.mention_hidden_size,
                hidden_dim=self.mention_hidden_size,
            )


        self.cluster_model_hidden_size = 768
        self.cluster_model_num_layers = 1
        self.cluster_model_config = DistilBertConfig(hidden_size=self.cluster_model_hidden_size)
        self.cluster_model = DistilBertModel(self.cluster_model_config).to(self.encoder.device)
        self.cluster_model.transformer.layer = self.cluster_model.transformer.layer[: self.cluster_model_num_layers]
        self.cluster_model.embeddings.word_embeddings = None
        self.cluster_transformer = attention(model=self.cluster_model, representation=self.cluster_representation)

        self.antecedent_coref_classifier = RepresentationLayer(
                type=self.representation_layer_type,  # fullyconnected
                input_dim=self.cluster_model_hidden_size,
                output_dim=self.cluster_model_hidden_size,
                hidden_dim=self.cluster_model_hidden_size,
            )

    def reset_parameters(self) -> None:
        W = [
            self.antecedent_s2s_all_weights,
            self.antecedent_e2e_all_weights,
            self.antecedent_s2e_all_weights,
            self.antecedent_e2s_all_weights,
        ]

        B = [
            self.antecedent_s2s_all_biases,
            self.antecedent_e2e_all_biases,
            self.antecedent_s2e_all_biases,
            self.antecedent_e2s_all_biases,
        ]

        for w, b in zip(W, B):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(b, -bound, bound)

    # takes last_hidden_states, eos_mask, ground truth and stage
    def eos_mention_extraction(self, lhs, mask, gold_mentions, gold_starts, stage):
        start_idxs = []
        mention_idxs = []
        start_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        mention_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)

        if gold_starts != None:
            gold_starts = gold_starts.unsqueeze(0)
            gold_mentions = gold_mentions.unsqueeze(0)
        eos_mask = mask.unsqueeze(0)
        for bidx in range(0, lhs.shape[0]):
            lhs_batch = lhs[bidx]  # SEQ_LEN X HIDD_DIM
            eos_mask_batch = eos_mask[bidx]  # SEQ_LEN X SEQ_LEN

            # compute start logits
            start_logits_batch = self.start_token_classifier(lhs_batch).squeeze(-1)  # SEQ_LEN

            if stage != "test":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(start_logits_batch, gold_starts[bidx])

                # accumulate loss
                start_loss = start_loss + loss

            # compute start positions
            start_idxs_batch = ((torch.sigmoid(start_logits_batch) > 0.5)).nonzero(as_tuple=False).squeeze(-1)

            start_idxs.append(start_idxs_batch.detach().clone())
            # in training, use gold starts to learn to extract mentions, inference use predicted ones
            if stage == "train":
                start_idxs_batch = (
                    ((torch.sigmoid(gold_starts[bidx]) > 0.5)).nonzero(as_tuple=False).squeeze(-1)
                )  # NUM_GOLD_STARTS

            # contains all possible start end indices pairs, i.e. for all starts, all possible ends looking at EOS index
            possibles_start_end_idxs = (eos_mask_batch[start_idxs_batch] == 1).nonzero(as_tuple=False)  # STARTS x 2

            # this is to have reference respect to original positions
            possibles_start_end_idxs[:, 0] = start_idxs_batch[possibles_start_end_idxs[:, 0]]

            possible_start_idxs = possibles_start_end_idxs[:, 0]
            possible_end_idxs = possibles_start_end_idxs[:, 1]

            # extract start and end hidden states
            starts_hidden_states = lhs_batch[possible_end_idxs]  # start
            ends_hidden_states = lhs_batch[possible_start_idxs]  # end

            # concatenation of start to end representations created using a representation layer
            s2e_representations = torch.cat(
                (
                    self.start_token_representation(starts_hidden_states),
                    self.end_token_representation(ends_hidden_states),
                ),
                dim=-1,
            )

            # classification of mentions
            s2e_logits = self.start_end_classifier(s2e_representations).squeeze(-1)

            # mention_start_idxs and mention_end_idxs
            mention_idxs.append(possibles_start_end_idxs[torch.sigmoid(s2e_logits) > 0.5].detach().clone())

            if s2e_logits.shape[0] != 0 and stage != "test":
                if gold_mentions != None:
                    mention_loss_batch = torch.nn.functional.binary_cross_entropy_with_logits(
                        s2e_logits,
                        gold_mentions[bidx][possible_start_idxs, possible_end_idxs],
                    )
                    mention_loss = mention_loss + mention_loss_batch

        return (start_idxs, mention_idxs, start_loss, mention_loss)

    def _get_all_labels(self, clusters_labels, categories_masks):
        batch_size, max_k, _ = clusters_labels.size()

        categories_labels = clusters_labels.unsqueeze(1).repeat(1, self.num_cats, 1, 1) * categories_masks
        all_labels = categories_labels

        return all_labels

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k), device="cpu")
        all_clusters_cpu = all_clusters.cpu().numpy()

        for b, (starts, ends, gold_clusters) in enumerate(
            zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)
        ):
            gold_clusters = self.extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.encoder.device)
        # new_cluster_labels = new_cluster_labels.to("cpu")
        return new_cluster_labels

    def _get_cluster_labels_after_pruning3(self, span_starts, span_ends, all_clusters, wrong_idxs):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k + 1, max_k), device="cpu")
        all_clusters_cpu = all_clusters.cpu().numpy()
        wrong_idxs = [(a.item(), b.item()) for a, b in wrong_idxs]
        for b, (starts, ends, gold_clusters) in enumerate(
            zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)
        ):
            gold_clusters = self.extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    if (start, end) not in wrong_idxs:
                        print(start, end)
                    else:
                        new_cluster_labels[b, -1, i] = 1
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
                # if (start, end) in wrong_idxs:
                #     new_cluster_labels[b, -1, i] = 1
        new_cluster_labels = new_cluster_labels.to(self.encoder.device)
        # new_cluster_labels = new_cluster_labels.to("cpu")
        return new_cluster_labels

    def _coreff(self, coref_idx, all_clusters):

        max_k = len([x for xx in coref_idx for x in xx])
        new_cluster_labels = torch.zeros((1, max_k, max_k), device="cpu")

        all_clusters_cpu = all_clusters.cpu().numpy()
        temp = [x for xx in coref_idx for x in xx]

        gold_clusters = self.extract_clusters(all_clusters_cpu)
        gold_clusters3 = [[[s for s in c if s[0] == key] for key in list(set([k[0] for k in c]))] for c in gold_clusters]

        for i, cluster in enumerate(temp):
            for temp_clust in gold_clusters3:
                for window_clust in temp_clust:
                    if len(set(tuple(cluster)) & set(tuple(window_clust))) == len(cluster):
                        for j, cluster3 in enumerate(temp[:i]):
                            for window_clust3 in temp_clust:
                                if len(set(tuple(cluster3)) & set(tuple(window_clust3))) == len(cluster3):
                                    new_cluster_labels[0, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.encoder.device)
        return new_cluster_labels

    def _coreff3(self, coref_idx, all_clusters, wrong_clusters):

        max_k = len([x for xx in coref_idx for x in xx])
        new_cluster_labels = torch.zeros((1, max_k + 1, max_k), device="cpu")

        all_clusters_cpu = all_clusters.cpu().numpy()
        temp = [x for xx in coref_idx for x in xx]

        gold_clusters = self.extract_clusters(all_clusters_cpu)
        gold_clusters3 = [[[s for s in c if s[0] == key] for key in list(set([k[0] for k in c]))] for c in gold_clusters]

        for i, cluster in enumerate(temp):
            for c in wrong_clusters:
                if tuple(cluster) in [tuple(cc) for cc in c]:
                    new_cluster_labels[0, -1, i] = 1
                    continue
            for temp_clust in gold_clusters3:
                for window_clust in temp_clust:
                    if len(set(tuple(cluster)) & set(tuple(window_clust))) == len(cluster):
                        for j, cluster3 in enumerate(temp[:i]):
                            for window_clust3 in temp_clust:
                                if len(set(tuple(cluster3)) & set(tuple(window_clust3))) == len(cluster3):
                                    new_cluster_labels[0, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.encoder.device)
        return new_cluster_labels

    def mes_span_clustering(
        self, mention_start_reps, mention_end_reps, mention_start_idxs, mention_end_idxs, gold, stage, mask, add, sing
    ):
        if mention_start_reps[0].shape[0] == 0:
            return torch.tensor([0.0], requires_grad=True, device=self.encoder.device), []
        coref_logits = self._calc_coref_logits(mention_start_reps, mention_end_reps)
        coref_logits = coref_logits[0] * mask[0]
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits]).unsqueeze(0)

        # # # batched
        # coref_logits = self._calc_coref_logits_batched(mention_start_reps, mention_end_reps)
        # coref_logits = coref_logits[0] * mask[0]
        # coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        # coref_logits = coref_logits.unsqueeze(0)

        # coref_logits_replica, logits_replica, biases_replica = self._calc_coref_logits(mention_start_reps, mention_end_reps)
        # coref_logits_b, logits_b, biases_b = self._calc_coref_logits_batched(mention_start_reps, mention_end_reps)
        # coref_logits_b_replica, logits_b_replica, biases_b_replica = self._calc_coref_logits_batched(
        #     mention_start_reps, mention_end_reps
        # )
        # logits = logits[0] * mask[0]
        # biases = biases[0] * mask[0]
        # logits_b = logits_b[0] * mask[0]
        # biases_b = biases_b[0] * mask[0]
        # coref_logits_replica = coref_logits_replica[0] * mask[0]
        # coref_logits_b = coref_logits_b[0] * mask[0]
        # coref_logits_b_replica = coref_logits_b_replica[0] * mask[0]
        # coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits]).unsqueeze(0)
        # coref_logits_replica = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits_replica]).unsqueeze(0)
        # coref_logits_b = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits_b]).unsqueeze(0)
        # coref_logits_b_replica = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits_b_replica]).unsqueeze(0)
        if stage == "train":
            labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, gold)
            all_labels = self._get_all_labels(labels, mask)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, all_labels) / 10

        # labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, gold)
        # all_labels = self._get_all_labels(labels, mask)
        # coref_logits = all_labels
        coref_logits = coref_logits.sum(dim=1)
        doc, m2a, singletons = self.create_mention_to_antecedent_singletons(mention_start_idxs, mention_end_idxs, coref_logits)
        if not sing:
            singletons = []
        coreferences = self.create_clusters(m2a, singletons, add)
        return coreference_loss, coreferences

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_cats, self.token_hidden_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # bnkf/bnlg

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s

        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = (
            top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
        )  # [batch_size, max_k, max_k]
        return coref_logits

    def _calc_coref_logits3(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s
        temp = torch.cat(
            (
                self.antecedent_s2s_classifier(top_k_start_coref_reps),
                torch.ones((1, 1, top_k_start_coref_reps.shape[-1])).to(self.encoder.device),
            ),
            dim=1,
        )  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = torch.cat(
            (
                self.antecedent_e2e_classifier(top_k_end_coref_reps),
                torch.ones((1, 1, top_k_end_coref_reps.shape[-1])).to(self.encoder.device),
            ),
            dim=1,
        )  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = torch.cat(
            (
                self.antecedent_s2e_classifier(top_k_start_coref_reps),
                torch.ones((1, 1, top_k_start_coref_reps.shape[-1])).to(self.encoder.device),
            ),
            dim=1,
        )  # [batch_size, max_k, dim]

        top_k_s2e_coref_logits = torch.matmul(temp, top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s

        temp = torch.cat(
            (
                self.antecedent_e2s_classifier(top_k_end_coref_reps),
                torch.ones((1, 1, top_k_end_coref_reps.shape[-1])).to(self.encoder.device),
            ),
            dim=1,
        )  # [batch_size, max_k, dim]

        top_k_e2s_coref_logits = torch.matmul(temp, top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = (
            top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
        )  # [batch_size, max_k, max_k]
        return coref_logits

    def _calc_coreference_logits(self, coref_hs):
        # s2s
        if self.cluster_representation == "s2e":
            start_hs, end_hs = coref_hs[0], coref_hs[1]
            temp = self.s2s_t_classifier(start_hs)
            # [batch_size, max_k, dim]
            top_k_s2s_coref_logits = torch.matmul(temp, start_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # e2e
            temp = self.e2e_t_classifier(end_hs)
            # [batch_size, max_k, dim]
            top_k_e2e_coref_logits = torch.matmul(temp, end_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # s2e
            temp = self.s2e_t_classifier(start_hs)
            # [batch_size, max_k, dim]

            top_k_s2e_coref_logits = torch.matmul(temp, end_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # e2s
            # e2s_t_classifier // antecedent_e2s_classifier
            temp = self.e2s_t_classifier(end_hs)
            # [batch_size, max_k, dim]

            top_k_e2s_coref_logits = torch.matmul(temp, start_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # sum all terms
            coref_logits = (
                top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
            )  # [batch_size, max_k, max_k]
        else:
            temp = self.antecedent_coref_classifier(coref_hs)
            coref_logits = torch.matmul(temp, coref_hs.permute([0, 2, 1]))
        return coref_logits

    def _calc_coreference_logits_s(self, coref_hs):
        # s2s
        if self.cluster_representation == "soft-dot":

            temp = self.antecedent_coref_classifier(coref_hs)

            coref_logits = torch.matmul(temp, coref_hs.permute([0, 2, 1]))
        elif self.cluster_representation == "soft-mc":
            start_hs, end_hs = coref_hs[0], coref_hs[1]
            temp = self.s2s_tmc_classifier(start_hs)

            top_k_s2s_coref_logits = torch.matmul(temp, start_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # e2e
            temp = self.e2e_tmc_classifier(end_hs)

            top_k_e2e_coref_logits = torch.matmul(temp, end_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # s2e
            temp = self.s2e_tmc_classifier(start_hs)

            top_k_s2e_coref_logits = torch.matmul(temp, end_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # e2s

            temp = self.e2s_tmc_classifier(end_hs)

            top_k_e2s_coref_logits = torch.matmul(temp, start_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # sum all terms
            coref_logits = (
                top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
            )  # [batch_size, max_k, max_k]
        return coref_logits

    def _calc_coreference_logits3(self, coref_hs):
        # s2s
        if self.cluster_representation == "s2e":
            start_hs, end_hs = coref_hs[0], coref_hs[1]
            temp = torch.cat(
                (
                    self.s2s_t_classifier(start_hs),
                    torch.ones((1, 1, start_hs.shape[-1])).to(self.encoder.device),
                ),
                dim=1,
            )  # [batch_size, max_k, dim]
            top_k_s2s_coref_logits = torch.matmul(temp, start_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # e2e
            temp = torch.cat(
                (
                    self.e2e_t_classifier(end_hs),
                    torch.ones((1, 1, end_hs.shape[-1])).to(self.encoder.device),
                ),
                dim=1,
            )  # [batch_size, max_k, dim]
            top_k_e2e_coref_logits = torch.matmul(temp, end_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # s2e
            temp = torch.cat(
                (
                    self.s2e_t_classifier(start_hs),
                    torch.ones((1, 1, start_hs.shape[-1])).to(self.encoder.device),
                ),
                dim=1,
            )  # [batch_size, max_k, dim]

            top_k_s2e_coref_logits = torch.matmul(temp, end_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # e2s

            temp = torch.cat(
                (
                    self.antecedent_e2s_classifier(end_hs),
                    torch.ones((1, 1, end_hs.shape[-1])).to(self.encoder.device),
                ),
                dim=1,
            )  # [batch_size, max_k, dim]

            top_k_e2s_coref_logits = torch.matmul(temp, start_hs.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

            # sum all terms
            coref_logits = (
                top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
            )  # [batch_size, max_k, max_k]
        else:
            # s2s
            temp = torch.cat(
                (
                    self.antecedent_coref_classifier(coref_hs),
                    torch.ones((1, 1, coref_hs.shape[-1])).to(self.encoder.device),
                ),
                dim=1,
            )  # [batch_size, max_k, dim]
        coref_logits = torch.matmul(temp, coref_hs.permute([0, 2, 1]))
        return coref_logits

    def _calc_coref_logits_batched(self, top_k_start_coref_reps, top_k_end_coref_reps):
        max_length = 10
        size = top_k_start_coref_reps.shape[1]
        if max_length > size:
            all_starts = self.transpose_for_scores(self.coref_start_all_mlps(top_k_start_coref_reps))
            all_ends = self.transpose_for_scores(self.coref_end_all_mlps(top_k_end_coref_reps))

            logits = (
                torch.einsum("bnkf, nfg, bnlg -> bnkl", all_starts, self.antecedent_s2s_all_weights, all_starts)
                + torch.einsum("bnkf, nfg, bnlg -> bnkl", all_ends, self.antecedent_e2e_all_weights, all_ends)
                + torch.einsum("bnkf, nfg, bnlg -> bnkl", all_starts, self.antecedent_s2e_all_weights, all_ends)
                + torch.einsum("bnkf, nfg, bnlg -> bnkl", all_ends, self.antecedent_e2s_all_weights, all_starts)
            ).to("cpu")

            biases = (
                torch.einsum("bnkf, nf -> bnk", all_starts, self.antecedent_s2s_all_biases).unsqueeze(-2)
                + torch.einsum("bnkf, nf -> bnk", all_ends, self.antecedent_e2e_all_biases).unsqueeze(-2)
                + torch.einsum("bnkf, nf -> bnk", all_ends, self.antecedent_s2e_all_biases).unsqueeze(-2)
                + torch.einsum("bnkf, nf -> bnk", all_starts, self.antecedent_e2s_all_biases).unsqueeze(-2)
            ).to("cpu")
        else:
            prev = 0
            logits = []
            biases = []
            # all_starts = self.transpose_for_scores(self.coref_start_all_mlps(top_k_start_coref_reps))
            # all_ends = self.transpose_for_scores(self.coref_end_all_mlps(top_k_end_coref_reps))

            indices = list(range(max_length, size, max_length))
            if len(indices) != 0 and indices[-1] != size:
                indices.append(size)
            for index in indices:

                # mention_batch_starts = all_starts[0][:, prev:index].unsqueeze(0)
                # mention_batch_ends = all_ends[0][:, prev:index].unsqueeze(0)
                # antecedent_batch_starts = all_starts[0][:, :index].unsqueeze(0)
                # antecedent_batch_ends = all_ends[0][:, :index].unsqueeze(0)
                mention_batch_starts = self.transpose_for_scores(
                    self.coref_start_all_mlps(top_k_start_coref_reps[0][prev:index].unsqueeze(0))
                )
                mention_batch_ends = self.transpose_for_scores(
                    self.coref_end_all_mlps(top_k_end_coref_reps[0][prev:index].unsqueeze(0))
                )
                antecedent_batch_starts = self.transpose_for_scores(
                    self.coref_start_all_mlps(top_k_start_coref_reps[0][:index].unsqueeze(0))
                )
                antecedent_batch_ends = self.transpose_for_scores(
                    self.coref_end_all_mlps(top_k_end_coref_reps[0][:index].unsqueeze(0))
                )

                pad = torch.zeros(mention_batch_starts.shape[0], mention_batch_starts.shape[1], index - prev, size)
                logit = (
                    torch.einsum(
                        "bnkf, nfg, bnlg -> bnkl",
                        mention_batch_starts,
                        self.antecedent_s2s_all_weights,
                        antecedent_batch_starts,
                    )
                    + torch.einsum(
                        "bnkf, nfg, bnlg -> bnkl",
                        mention_batch_ends,
                        self.antecedent_e2e_all_weights,
                        antecedent_batch_ends,
                    )
                    + torch.einsum(
                        "bnkf, nfg, bnlg -> bnkl",
                        mention_batch_starts,
                        self.antecedent_s2e_all_weights,
                        antecedent_batch_ends,
                    )
                    + torch.einsum(
                        "bnkf, nfg, bnlg -> bnkl",
                        mention_batch_ends,
                        self.antecedent_e2s_all_weights,
                        antecedent_batch_starts,
                    )
                )

                pad[:, :, :, :index] = logit

                logits.append(pad)

                b = (
                    torch.einsum("bnkf, nf -> bnk", mention_batch_starts, self.antecedent_s2s_all_biases).unsqueeze(-2)
                    + torch.einsum("bnkf, nf -> bnk", mention_batch_ends, self.antecedent_e2e_all_biases).unsqueeze(-2)
                    + torch.einsum("bnkf, nf -> bnk", mention_batch_ends, self.antecedent_s2e_all_biases).unsqueeze(-2)
                    + torch.einsum("bnkf, nf -> bnk", mention_batch_starts, self.antecedent_e2s_all_biases).unsqueeze(-2)
                )

                biases.append(b)
                prev = index
            # logits = torch.cat(logits, dim=2).to(self.encoder.device)
            # biases = torch.cat(biases, dim=3)
            logits = torch.cat(logits, dim=2).to("cpu")
            biases = torch.cat(biases, dim=3).to("cpu")
        return logits + biases

    def _get_categories_labels(self, tokens, subtoken_map, new_token_map, span_starts, span_ends):
        max_k = span_starts.shape[0]

        doc_spans = []
        for start, end in zip(span_starts, span_ends):
            token_indices = [new_token_map[0][idx] for idx in set(subtoken_map[0][start : end + 1]) - {None}]
            span = {tokens[0][idx].lower() for idx in token_indices if idx is not None}
            pronoun_id = get_pronoun_id(span)
            doc_spans.append((span - STOPWORDS, pronoun_id))

        categories_labels = np.zeros((max_k, max_k)) - 1
        for i in range(max_k):
            for j in list(range(max_k))[:i]:
                categories_labels[i, j] = get_category_id(doc_spans[i], doc_spans[j])

        categories_labels = torch.tensor(categories_labels, device=self.encoder.device).unsqueeze(0)
        # categories_labels = torch.tensor(categories_labels, device="cpu").unsqueeze(0)
        categories_masks = [categories_labels == cat_id for cat_id in range(self.num_cats - 1)] + [categories_labels != -1]
        categories_masks = torch.stack(categories_masks, dim=1).int()
        return categories_labels, categories_masks

    def create_mention_to_antecedent_singletons(self, span_starts, span_ends, coref_logits):
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        bs, n_spans, _ = coref_logits.shape
        # coref_logits = coref_logits.detach().cpu()
        # long distance regularization
        # a = torch.sigmoid(coref_logits)
        # m1 = torch.arange(coref_logits.shape[1], device=a.device).unsqueeze(0).repeat(coref_logits.shape[1], 1)
        # m2 = m1.transpose(0, 1)
        # m = (torch.ones_like(coref_logits[0], device=a.device) + (m2 - m1 - 1) / 1000).tril().fill_diagonal_(0).unsqueeze(0)
        # no_ant = 1 - torch.sum(a - a * m > 0.5, dim=-1).bool().float()
        # coref_logits = torch.cat((coref_logits - coref_logits * m, no_ant.unsqueeze(-1)), dim=-1)
        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        span_starts = span_starts.detach().cpu()
        span_ends = span_ends.detach().cpu()
        max_antecedents = coref_logits.argmax(axis=-1).detach().cpu()
        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

        antecedent_indices = max_antecedents[max_antecedents < n_spans]
        span_indices = np.stack([span_starts.detach().cpu(), span_ends.detach().cpu()], axis=-1)

        mentions = span_indices[doc_indices, mention_indices]
        antecedents = span_indices[doc_indices, antecedent_indices]
        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        sing_indices = np.zeros_like(len(np.setdiff1d(non_mentions, antecedent_indices)))
        singletons = span_indices[sing_indices, np.setdiff1d(non_mentions, antecedent_indices)]

        # mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1 and len(antecedents.shape) == 1:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=0)
        else:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1:
            mention_to_antecedent = [mention_to_antecedent]

        if len(singletons.shape) == 1:
            singletons = [singletons]

        return doc_indices, mention_to_antecedent, singletons

    def create_mention_to_antecedent_singletons3(self, span_starts, span_ends, coref_logits):
        span_starts = span_starts.unsqueeze(0)
        span_ends = span_ends.unsqueeze(0)
        bs, n_spans, _ = coref_logits.shape
        n_spans = n_spans - 1
        # coref_logits = coref_logits.detach().cpu()
        # long distance regularization
        # a = torch.sigmoid(coref_logits)
        # m1 = torch.arange(coref_logits.shape[1], device=a.device).unsqueeze(0).repeat(coref_logits.shape[1], 1)
        # m2 = m1.transpose(0, 1)
        # m = (torch.ones_like(coref_logits[0], device=a.device) + (m2 - m1 - 1) / 1000).tril().fill_diagonal_(0).unsqueeze(0)
        # no_ant = 1 - torch.sum(a - a * m > 0.5, dim=-1).bool().float()
        # coref_logits = torch.cat((coref_logits - coref_logits * m, no_ant.unsqueeze(-1)), dim=-1)
        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        span_starts = span_starts.detach().cpu()
        span_ends = span_ends.detach().cpu()
        max_antecedents = coref_logits[:, :-1].argmax(axis=-1).detach().cpu()
        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]
        wrong_mentions_indices = (
            (torch.sigmoid(coref_logits[0][-1][:-1]).detach().cpu() > 0.5).nonzero(as_tuple=False).squeeze(-1)
        )

        antecedent_indices = max_antecedents[max_antecedents < n_spans]
        span_indices = np.stack([span_starts.detach().cpu(), span_ends.detach().cpu()], axis=-1)

        mentions = span_indices[doc_indices, mention_indices]
        antecedents = span_indices[doc_indices, antecedent_indices]
        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]
        wrong_mentions = span_indices[0, wrong_mentions_indices]
        # singletons = np.setdiff1d(np.setdiff1d(non_mentions, antecedent_indices), wrong_mentions)
        a = np.setdiff1d(non_mentions, antecedent_indices)
        sing_indices = np.zeros_like(len(np.setdiff1d(a, wrong_mentions)))

        singletons = span_indices[sing_indices, np.setdiff1d(a, wrong_mentions)]

        # mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1 and len(antecedents.shape) == 1:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=0)
        else:
            mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

        if len(mentions.shape) == 1:
            mention_to_antecedent = [mention_to_antecedent]

        if len(singletons.shape) == 1:
            singletons = [singletons]

        return doc_indices, mention_to_antecedent, singletons

    def create_temp_to_antecedent_singletons(self, tempp, coref_logits):

        bs, n_spans, _ = coref_logits.shape
        # coref_logits = coref_logits.detach().cpu()
        # long distance regularization
        # a = torch.sigmoid(coref_logits)
        # m1 = torch.arange(coref_logits.shape[1], device=a.device).unsqueeze(0).repeat(coref_logits.shape[1], 1)
        # m2 = m1.transpose(0, 1)
        # m = (torch.ones_like(coref_logits[0], device=a.device) + (m2 - m1 - 1) / 1000).tril().fill_diagonal_(0).unsqueeze(0)
        # no_ant = 1 - torch.sum(a - a * m > 0.5, dim=-1).bool().float()
        # coref_logits = torch.cat((coref_logits - coref_logits * m, no_ant.unsqueeze(-1)), dim=-1)
        # same_window = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]

        # add = torch.ones_like(coref_logits[0]).triu_().fill_diagonal_(1)
        # add = add.unsqueeze(0) * -5
        # coref_logits = coref_logits + add

        # no_ant = -4 - torch.sum(coref_logits > -5, dim=-1).bool().float()

        t = coref_logits
        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        # tempp = tempp.detach().cpu()
        # span_starts = span_starts.detach().cpu()
        # span_ends = span_ends.detach().cpu()

        # torch.ones_like(coref_logits).triu.fill_diagonal(1)*-1

        max_antecedents = coref_logits.argmax(axis=-1).detach().cpu()

        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

        antecedent_indices = max_antecedents[max_antecedents < n_spans]

        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        if False:
            singleton_indices = [i for i in non_mentions if i not in mention_indices and i not in antecedent_indices]

            for i in singleton_indices:
                if i < 1:
                    continue
                if t[0][i][torch.argmax(t[0][i][t[0][i] != 0])] > -10:
                    max_antecedents[0][i] = torch.argmax(t[0][i][t[0][i] != 0])

            doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
            # indices where antecedent is not null.
            mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

            antecedent_indices = max_antecedents[max_antecedents < n_spans]

            non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        tempp = [item for sublist in tempp for item in sublist]

        cluster = [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] != tempp[t2][0][0]]
        # singletons.extend(
        #     [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] == tempp[t2][0][0]]
        # )

        clusters = [tuple(tempp[t2]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] != tempp[t2][0][0]]

        singletons = [tuple(tempp[i]) for i in non_mentions if i not in mention_indices and i not in antecedent_indices]

        # singletons.extend(
        #     [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] == tempp[t2][0][0]]
        # )
        for t in [tuple(t1) for t1 in tempp if tuple(t1) not in cluster and tuple(t1) not in clusters]:
            singletons.append(t)
        # singletons = [elem for elem in singletons if elem not in cluster and elem not in clusters]
        singletons = list(sorted(set(tuple(singletons))))
        return {tuple(c1): tuple(c2) for c1, c2 in zip(cluster, clusters) if c1[0][0] != c2[0][0]}, singletons

    def create_temp_to_antecedent_singletonst(self, tempp, coref_logits):

        bs, n_spans, _ = coref_logits.shape
        # coref_logits = coref_logits.detach().cpu()
        # long distance regularization
        # a = torch.sigmoid(coref_logits)
        # m1 = torch.arange(coref_logits.shape[1], device=a.device).unsqueeze(0).repeat(coref_logits.shape[1], 1)
        # m2 = m1.transpose(0, 1)
        # m = (torch.ones_like(coref_logits[0], device=a.device) + (m2 - m1 - 1) / 1000).tril().fill_diagonal_(0).unsqueeze(0)
        # no_ant = 1 - torch.sum(a - a * m > 0.5, dim=-1).bool().float()
        # coref_logits = torch.cat((coref_logits - coref_logits * m, no_ant.unsqueeze(-1)), dim=-1)
        # same_window = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]

        # add = torch.ones_like(coref_logits[0]).triu_().fill_diagonal_(1)
        # add = add.unsqueeze(0) * -5
        # coref_logits = coref_logits + add

        # no_ant = -4 - torch.sum(coref_logits > -5, dim=-1).bool().float()

        t = coref_logits
        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        # tempp = tempp.detach().cpu()
        # span_starts = span_starts.detach().cpu()
        # span_ends = span_ends.detach().cpu()

        # torch.ones_like(coref_logits).triu.fill_diagonal(1)*-1

        max_antecedents = coref_logits.argmax(axis=-1).detach().cpu()

        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

        antecedent_indices = max_antecedents[max_antecedents < n_spans]

        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        if True:
            singleton_indices = [i for i in non_mentions if i not in mention_indices and i not in antecedent_indices]

            for i in singleton_indices:
                if i < 1:
                    continue
                if -100 in coref_logits[0][i]:
                    continue
                max_antecedents[0][i] = torch.argmax(t[0][i][t[0][i] != 0])

            doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
            # indices where antecedent is not null.
            mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

            antecedent_indices = max_antecedents[max_antecedents < n_spans]

            non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]

        tempp = [item for sublist in tempp for item in sublist]

        cluster = [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] != tempp[t2][0][0]]
        # singletons.extend(
        #     [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] == tempp[t2][0][0]]
        # )

        clusters = [tuple(tempp[t2]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] != tempp[t2][0][0]]

        singletons = [tuple(tempp[i]) for i in non_mentions if i not in mention_indices and i not in antecedent_indices]

        # singletons.extend(
        #     [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] == tempp[t2][0][0]]
        # )
        for t in [tuple(t1) for t1 in tempp if tuple(t1) not in cluster and tuple(t1) not in clusters]:
            singletons.append(t)
        # singletons = [elem for elem in singletons if elem not in cluster and elem not in clusters]
        singletons = list(sorted(set(tuple(singletons))))
        return {tuple(c1): tuple(c2) for c1, c2 in zip(cluster, clusters) if c1[0][0] != c2[0][0]}, singletons

    def create_temp_to_antecedent_singletons3(self, tempp, coref_logits):

        bs, n_spans, _ = coref_logits.shape
        n_spans = n_spans - 1
        # coref_logits = coref_logits.detach().cpu()
        # long distance regularization
        # a = torch.sigmoid(coref_logits)
        # m1 = torch.arange(coref_logits.shape[1], device=a.device).unsqueeze(0).repeat(coref_logits.shape[1], 1)
        # m2 = m1.transpose(0, 1)
        # m = (torch.ones_like(coref_logits[0], device=a.device) + (m2 - m1 - 1) / 1000).tril().fill_diagonal_(0).unsqueeze(0)
        # no_ant = 1 - torch.sum(a - a * m > 0.5, dim=-1).bool().float()
        # coref_logits = torch.cat((coref_logits - coref_logits * m, no_ant.unsqueeze(-1)), dim=-1)
        no_ant = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # same_window = 1 - torch.sum(torch.sigmoid(coref_logits) > 0.5, dim=-1).bool().float()
        # [batch_size, max_k, max_k + 1]
        coref_logits = torch.cat((coref_logits, no_ant.unsqueeze(-1)), dim=-1)

        # tempp = tempp.detach().cpu()
        # span_starts = span_starts.detach().cpu()
        # span_ends = span_ends.detach().cpu()
        max_antecedents = coref_logits[:, :-1].argmax(axis=-1).detach().cpu()
        doc_indices = np.nonzero(max_antecedents < n_spans)[:, 0]
        # indices where antecedent is not null.
        mention_indices = np.nonzero(max_antecedents < n_spans)[:, 1]

        antecedent_indices = max_antecedents[max_antecedents < n_spans]
        wrong_mentions_indices = (
            (torch.sigmoid(coref_logits[0][-1][:-1]).detach().cpu() > 0.5).nonzero(as_tuple=False).squeeze(-1)
        )
        non_mentions = np.nonzero(max_antecedents == n_spans)[:, 1]
        tempp = [item for sublist in tempp for item in sublist]

        singletons = [tuple(tempp[i]) for i in non_mentions if i not in mention_indices and i not in antecedent_indices]

        cluster = [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] != tempp[t2][0][0]]

        # singletons.extend(
        #     [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] == tempp[t2][0][0]]
        # )

        clusters = [tuple(tempp[t2]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] != tempp[t2][0][0]]
        ttttttttt = [tuple(tempp[t1]) for t1 in wrong_mentions_indices]

        # singletons.extend(
        #     [tuple(tempp[t1]) for t1, t2 in zip(mention_indices, antecedent_indices) if tempp[t1][0][0] == tempp[t2][0][0]]
        # )
        for t in [tuple(t1) for t1 in tempp if tuple(t1) not in cluster and tuple(t1) not in clusters]:
            singletons.append(t)
        singletons = [t for t in singletons if t not in ttttttttt]
        # singletons = [elem for elem in singletons if elem not in cluster and elem not in clusters]
        singletons = list(sorted(set(tuple(singletons))))
        return {tuple(c1): tuple(c2) for c1, c2 in zip(cluster, clusters) if c1[0][0] != c2[0][0]}, singletons

    def create_clusters(self, m2a, singletons, add):
        # Note: mention_to_antecedent is a numpy array
        if add != None:
            clusters = add
            mention_to_cluster = {m: i for i, c in enumerate(clusters) for m in c}
        else:
            clusters, mention_to_cluster = [], {}
        for mention, antecedent in m2a:
            mention, antecedent = tuple(mention), tuple(antecedent)
            if mention in mention_to_cluster and antecedent in mention_to_cluster:
                continue
            if mention in mention_to_cluster:
                cluster_idx = mention_to_cluster[mention]
                if antecedent not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(antecedent)
                    mention_to_cluster[antecedent] = cluster_idx
            elif antecedent in mention_to_cluster:
                cluster_idx = mention_to_cluster[antecedent]
                if mention not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(mention)
                    mention_to_cluster[mention] = cluster_idx
            else:
                cluster_idx = len(clusters)
                mention_to_cluster[mention] = cluster_idx
                mention_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, mention])

        clusters = [tuple(cluster) for cluster in clusters]
        # maybe order stuff?
        if len(singletons) != 0:
            clust = []
            while len(clusters) != 0 or len(singletons) != 0:
                if len(singletons) == 0:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
                elif len(clusters) == 0:
                    clust.append(tuple([tuple(singletons[0])]))
                    singletons = singletons[1:]
                elif singletons[0][0] < sorted(clusters[0], key=lambda x: x[0])[0][0]:
                    clust.append(tuple([tuple(singletons[0])]))
                    singletons = singletons[1:]
                else:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
            return clust
        return clusters

    def create_temp(self, c2a, singletons, add):
        # Note: mention_to_antecedent is a numpy array
        if add != None:
            clusters = add
            cluster_to_cluster = {m: i for i, c in enumerate(clusters) for m in c}
        else:
            clusters, cluster_to_cluster = [], {}
        for cluster, antecedent in c2a.items():
            cluster, antecedent = tuple(cluster), tuple(antecedent)
            if cluster in cluster_to_cluster and antecedent in cluster_to_cluster:
                continue
            if cluster in cluster_to_cluster:
                cluster_idx = cluster_to_cluster[cluster]
                if antecedent not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(antecedent)
                    cluster_to_cluster[antecedent] = cluster_idx
            elif antecedent in cluster_to_cluster:
                cluster_idx = cluster_to_cluster[antecedent]
                if cluster not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(cluster)
                    cluster_to_cluster[cluster] = cluster_idx
            else:
                cluster_idx = len(clusters)
                cluster_to_cluster[cluster] = cluster_idx
                cluster_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, cluster])

        clusters = list(sorted(set([tuple([x for xx in temp for x in xx]) for temp in clusters])))

        if len(singletons) != 0:
            clust = []
            while len(clusters) != 0 or len(singletons) != 0:
                if len(singletons) == 0:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
                elif len(clusters) == 0:
                    clust.append(singletons[0])
                    singletons = singletons[1:]
                elif singletons[0][0][0] + singletons[0][0][1] < clusters[0][0][0] + clusters[0][0][1]:
                    clust.append(singletons[0])
                    singletons = singletons[1:]
                else:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
            return clust
        return clusters

    def create_temp_ttttttt(self, c2na, c2a, singletons, add):
        # Note: mention_to_antecedent is a numpy array

        if add != None:
            clusters = add
            cluster_to_cluster = {m: i for i, c in enumerate(clusters) for m in c}
        else:
            clusters, cluster_to_cluster = [], {}
        for cluster, antecedent in c2a.items():
            cluster, antecedent = tuple(cluster), tuple(antecedent)
            if cluster in cluster_to_cluster and antecedent in cluster_to_cluster:
                continue
            if cluster in cluster_to_cluster:
                if sum([(cluster, c) in c2na for c in clusters[cluster_to_cluster[cluster]]]) > 0:
                    if cluster not in singletons:
                        singletons.append(cluster)
                    continue
            if antecedent in cluster_to_cluster:
                if sum([(antecedent, c) in c2na for c in clusters[cluster_to_cluster[antecedent]]]) > 0:
                    if antecedent not in singletons:
                        singletons.append(antecedent)
                    continue
            if cluster in cluster_to_cluster:
                cluster_idx = cluster_to_cluster[cluster]
                if antecedent not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(antecedent)
                    cluster_to_cluster[antecedent] = cluster_idx
            elif antecedent in cluster_to_cluster:
                cluster_idx = cluster_to_cluster[antecedent]
                if cluster not in clusters[cluster_idx]:
                    clusters[cluster_idx].append(cluster)
                    cluster_to_cluster[cluster] = cluster_idx
            else:
                cluster_idx = len(clusters)
                cluster_to_cluster[cluster] = cluster_idx
                cluster_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, cluster])

        clusters = list(sorted(set([tuple([x for xx in temp for x in xx]) for temp in clusters])))

        if len(singletons) != 0:
            clust = []
            while len(clusters) != 0 or len(singletons) != 0:
                if len(singletons) == 0:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
                elif len(clusters) == 0:
                    clust.append(singletons[0])
                    singletons = singletons[1:]
                elif singletons[0][0][0] + singletons[0][0][1] < clusters[0][0][0] + clusters[0][0][1]:
                    clust.append(singletons[0])
                    singletons = singletons[1:]
                else:
                    clust.append(clusters[0])
                    clusters = clusters[1:]
            return clust
        return clusters

    def extract_clusters(self, gold_clusters):
        if gold_clusters.shape[-1] == 3 and (
            self.cluster_representation == "soft-mc" or self.cluster_representation == "soft-dot"
        ):
            gold_clusters = [tuple(tuple([m[0] + m[1], m[0] + m[2]]) for m in c if (-1) not in m) for c in gold_clusters]
        else:
            gold_clusters = [tuple(tuple(m) for m in cluster if (-1) not in m) for cluster in gold_clusters]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        return gold_clusters

    def s2e_span_clustering(
        self, mention_start_reps, mention_end_reps, mention_start_idxs, mention_end_idxs, gold, stage, add, sing
    ):
        # coref_logits = self._calc_coref_logits_batched(mention_start_reps, mention_end_reps)
        coref_logits = self._calc_coref_logits(mention_start_reps, mention_end_reps)
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        gold = gold.squeeze(0)
        coref_logits = coref_logits[0]
        coref_logits.tril_().fill_diagonal_(0)
        coref_logits = coref_logits.unsqueeze(0)
        # coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        if stage == "train":
            labels = self._get_cluster_labels_after_pruning(mention_start_idxs, mention_end_idxs, gold)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, labels) / 10

        doc, m2a, singletons = self.create_mention_to_antecedent_singletons(mention_start_idxs, mention_end_idxs, coref_logits)
        if not sing:
            singletons = []
        coreferences = self.create_clusters(m2a, singletons, add)
        return coreference_loss, coreferences

    def s2e_span_clustering3(
        self, mention_start_reps, mention_end_reps, mention_start_idxs, mention_end_idxs, gold, stage, add, sing, wrong
    ):
        # coref_logits = self._calc_coref_logits_batched(mention_start_reps, mention_end_reps)
        coref_logits = self._calc_coref_logits3(mention_start_reps, mention_end_reps)
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        gold = gold.squeeze(0)
        coref_logits = coref_logits[0]
        coref_logits.tril_().fill_diagonal_(0)
        coref_logits = coref_logits.unsqueeze(0)
        # coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        if stage == "train":
            labels = self._get_cluster_labels_after_pruning3(mention_start_idxs, mention_end_idxs, gold, wrong)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, labels) / 10

        doc, m2a, singletons = self.create_mention_to_antecedent_singletons3(mention_start_idxs, mention_end_idxs, coref_logits)
        if not sing:
            singletons = []
        coreferences = self.create_clusters(m2a, singletons, add)
        return coreference_loss, coreferences

    def cluster_clustering_s(self, mention_reps, mention_idxs, gold, stage, add, sing):
        # coref_logits = self._calc_coref_logits_batched(mention_start_reps, mention_end_reps)
        coref_logits = self._calc_coreference_logits_s(mention_reps)
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        gold = gold.squeeze(0)

        coref_logits = coref_logits[0]
        coref_logits.tril_().fill_diagonal_(0)
        coref_logits = coref_logits.unsqueeze(0)

        # coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        if stage == "train":
            labels = self._get_cluster_labels_after_pruning(
                torch.tensor([m[0] for m in mention_idxs]), torch.tensor([m[1] for m in mention_idxs]), gold.unsqueeze(0)
            )
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, labels) / 3

        doc, m2a, singletons = self.create_mention_to_antecedent_singletons(
            torch.tensor([m[0] for m in mention_idxs]), torch.tensor([m[1] for m in mention_idxs]), coref_logits
        )
        coreferences = self.create_clusters(m2a, singletons, add)

        if not sing:
            coreferences = [i for i in coreferences if len(i) > 1]
        # return coreference_loss, coreferences
        return coreference_loss, coreferences

    def cluster_clustering(self, coref_reps, tempp, gold, stage, add, sing):
        # coref_logits = self._calc_coref_logits_batched(mention_start_reps, mention_end_reps)
        coref_logits = self._calc_coreference_logits(coref_reps)
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        
        if gold != None:
            gold = gold.squeeze(0)
        coref_logits = coref_logits[0]
        coref_logits.tril_().fill_diagonal_(0)
        coref_logits = coref_logits.unsqueeze(0)
        # coref_logits = torch.stack([matrix.tril().fill_diagonal_(0) for matrix in coref_logits])
        if stage == "train":
            labels = self._coreff(tempp, gold)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, labels) / 3
        # coref_logits = self._coreff(tempp, gold)
        c2a, singletons = self.create_temp_to_antecedent_singletons(tempp, coref_logits)
        c2na = []
        for clusters in tempp:
            for cluster in clusters:
                c2na.extend([(tuple(cluster), tuple(c)) for c in clusters if tuple(c) != tuple(cluster)])

        coreferences = self.create_temp(c2a, singletons, add)
        # ttt = [x for xx in tempp for x in xx]

        # for cluster in coreferences:
        #     idxs = []
        #     for tt in ttt:
        #         if len(set(tuple(tt)) & set(tuple(cluster))) == len(set(tuple(tt))):
        #             idxs.append(tt)
        #     for idx in idxs:
        #         for a, b in [(ttt.index(idx), ttt.index(c)) for c in idxs if (tuple(idx), tuple(c)) in c2na]:
        #             found = False
        #             ttttttt = 0
        #             if a > b:
        #                 while True:
        #                     ta = coref_logits[0][a][ttt.index(list(c2a[tuple(ttt[a])]))]
        #                     tttt = coref_logits[0][b][ttt.index(list(c2a[tuple(ttt[b])]))]
        #                     if ta < 0:
        #                         coref_logits[0][a][ttt.index(list(c2a[tuple(ttt[a])]))] = -100
        #                         ttttttt += 1
        #                     if tttt < 0:
        #                         coref_logits[0][b][ttt.index(list(c2a[tuple(ttt[b])]))] = -100
        #                         ttttttt += 1
        #                     if ttttttt > 3:
        #                         break
        #                     if tuple(c2a[tuple(ttt[a])]) in c2a:
        #                         a = ttt.index(list(c2a[tuple(ttt[a])]))
        #                     elif tuple(c2a[tuple(ttt[b])]) in c2a:
        #                         b = ttt.index(list(c2a[tuple(ttt[b])]))
        #                     else:
        #                         break

        # c2a, singletons = self.create_temp_to_antecedent_singletonst(tempp, coref_logits)
        # coreferences = self.create_temp(c2a, singletons, add)

        if not sing:
            coreferences = [i for i in coreferences if len(i) > 1]
        return coreference_loss, coreferences

    def cluster_clustering3(self, coref_reps, tempp, gold, stage, add, sing, wrong):
        coref_logits = self._calc_coreference_logits3(coref_reps)
        coreference_loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        gold = gold.squeeze(0)
        coref_logits = coref_logits[0]
        coref_logits.tril_().fill_diagonal_(0)
        coref_logits = coref_logits.unsqueeze(0)
        if stage == "train":
            labels = self._coreff3(tempp, gold, wrong)
            coreference_loss = torch.nn.functional.binary_cross_entropy_with_logits(coref_logits, labels) / 3

        c2a, singletons = self.create_temp_to_antecedent_singletons3(tempp, coref_logits)
        if not sing:
            singletons = [i for i in singletons if len(i) > 1]
        coreferences = self.create_temp(c2a, singletons, add)
        return coreference_loss, coreferences

    def forward(
        self,
        stage,
        input_ids,
        attention_mask,
        eos_mask=None,
        gold_starts=None,
        tokens=None,
        subtoken_map=None,
        new_token_map=None,
        gold_mentions=None,
        gold_clusters=None,
        add=None,
        singletons=False,
        full_clusters=None,
        temp=None,
    ):

        loss = torch.tensor([0.0], requires_grad=True, device=self.encoder.device)
        loss_dict = {}
        preds = {}

        if gold_starts == None:
            gold_starts = [None for i in range(len(eos_mask))]
            gold_mentions = [None for i in range(len(eos_mask))]
            gold_clusters = [None for i in range(len(eos_mask))]
        i = 0
        preds["start_idxs"] = []
        preds["mention_idxs"] = []
        preds["local_clusters"] = []
        preds["clusters"] = []
        preds["clusters_t"] = []
        coref_idxs = []
        coref_idxs_t = []
        coref_hs = []
        start_hs = []
        end_hs = []
        t3 = input_ids[0][:1]
        t33 = input_ids[0][-1:]
        loss_dict["start_loss"] = 0
        loss_dict["mention_loss"] = 0
        ttttttttt = []
        mention_t_hs = []
        cluster_t_hs = []
        mention_t_idxs = []
        for ids, am, mask, starts, mentions, clusters, t_tokens, t_subtoken_map, t_new_token_map in zip(
            input_ids, attention_mask, eos_mask, gold_starts, gold_mentions, gold_clusters, tokens, subtoken_map, new_token_map
        ):
            t = temp[i]
            if t != 0:
                ids = torch.cat((t3, ids), dim=-1)
                ids = torch.cat((ids, t33), dim=-1)
                am = torch.cat((t3, am), dim=-1)
                am = torch.cat((t3, am), dim=-1)
                last_hidden_states = self.encoder(input_ids=ids.unsqueeze(0), attention_mask=am.unsqueeze(0))[
                    "last_hidden_state"
                ][0][1:-1].unsqueeze(
                    0
                )  # B x S x TH
            else:
                last_hidden_states = self.encoder(input_ids=ids.unsqueeze(0), attention_mask=am.unsqueeze(0))["last_hidden_state"]
            (
                start_idxs,
                mention_idxs,
                start_loss,
                mention_loss,
            ) = self.eos_mention_extraction(
                lhs=last_hidden_states,
                mask=mask,
                gold_mentions=mentions,
                gold_starts=starts,
                stage=stage,
            )

            loss_dict["start_loss"] += start_loss
            startsss = [start_idxs[0] + temp[i]]
            preds["start_idxs"].extend([start.detach().cpu() for start in startsss])

            loss_dict["mention_loss"] += mention_loss
            m = [mention_idxs[0] + t]
            preds["mention_idxs"].extend([mention.detach().cpu() for mention in m])
            wrong = torch.tensor([])
            loss = loss + start_loss + mention_loss
            if stage == "train":
                if self.negatives:
                    mention_gold = (mentions == 1).nonzero(as_tuple=False)
                    ttt = []
                    for iii, tt in enumerate(mention_idxs[0].tolist()):
                        if tt not in mention_gold.tolist():
                            ttt.append(iii)
                    ttt = torch.tensor(ttt)
                    if ttt.shape[0] != 0:
                        difference = mention_idxs[0][ttt]
                        difference = difference[torch.randperm(difference.size()[0])]
                        mention_idxs = torch.cat((mention_gold, difference[:20]))
                        mention_idxs = mention_idxs[mention_idxs[:, 1].sort()[1]]
                        wrong = difference[:20]
                    else:
                        mention_idxs = (mentions == 1).nonzero(as_tuple=False)
                else:
                    mention_idxs = (mentions == 1).nonzero(as_tuple=False)

            elif stage == "test" and mentions != None:
                mention_idxs = (mentions == 1).nonzero(as_tuple=False)
            else:
                mention_idxs = mention_idxs[0]

            mention_start_idxs = mention_idxs[:, 0]
            mention_end_idxs = mention_idxs[:, 1]
            mentions_start_hidden_states = torch.index_select(last_hidden_states, 1, mention_start_idxs)
            mentions_end_hidden_states = torch.index_select(last_hidden_states, 1, mention_end_idxs)

            import gc

            gc.collect()
            torch.cuda.empty_cache()

            if mentions_start_hidden_states[0].shape[0] != 0:
                if self.negatives:
                    coreference_loss, coreferences = self.s2e_span_clustering3(
                        mentions_start_hidden_states,
                        mentions_end_hidden_states,
                        mention_start_idxs,
                        mention_end_idxs,
                        clusters.unsqueeze(0),
                        stage,
                        add,
                        1 == 1,
                        wrong,
                    )
                else:
                    if self.mes == True:
                        t_tokens = [t_tokens]
                        t_new_token_map = [t_new_token_map]
                        t_subtoken_map = [t_subtoken_map]
                        _, categories_masks = self._get_categories_labels(
                            t_tokens, t_subtoken_map, t_new_token_map, mention_start_idxs, mention_end_idxs
                        )
                        coreference_loss, coreferences = self.mes_span_clustering(
                            mentions_start_hidden_states,
                            mentions_end_hidden_states,
                            mention_start_idxs,
                            mention_end_idxs,
                            clusters,
                            stage,
                            categories_masks,
                            add,
                            True,
                        )
                    else:

                        coreference_loss, coreferences = self.s2e_span_clustering(
                            mentions_start_hidden_states,
                            mentions_end_hidden_states,
                            mention_start_idxs,
                            mention_end_idxs,
                            clusters.unsqueeze(0),
                            stage,
                            add,
                            1 == 1,
                        )

                loss = loss + coreference_loss

                if stage == "train":
                    golddd = unpad_gold_clusters(clusters)
                    if self.negatives:
                        cc = coreferences
                        tt = [
                            [(t, s[0], s[1]) for s in c]
                            for c in cc
                            if len([1 for e in golddd if len(set(e) & set(c)) > int(len(c) / 2)]) == 0
                        ]
                        ttttttttt.append(tt)
                        golddd.extend([c for c in cc if len([1 for e in golddd if len(set(e) & set(c)) > int(len(c) / 2)]) == 0])
                        sorted(golddd, key=lambda x: x[0][0])
                    coref_idxs_t.append(
                        [
                            [
                                (t, span[0], span[1])
                                for span in cluster
                                # if t_tokens[0][
                                #     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                                #         t_subtoken_map[0][span[0]]
                                #     ]
                                #     + 1
                                # ][0].lower()
                                # not in PRONOUNS_GROUPS
                                # and span[1] - span[0] < 10
                                # and t_tokens[0][
                                #     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                                #         t_subtoken_map[0][span[0]]
                                #     ]
                                #     + 1
                                # ][0][0].isupper()
                                # and t_tokens[0][
                                #     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                                #         t_subtoken_map[0][span[0]]
                                #     ]
                                #     + 1
                                # ][0][0].isupper()
                            ]
                            for cluster in golddd
                        ]
                    )
                    coref_idxs.append([[(t, span[0], span[1]) for span in cluster] for cluster in golddd])
                else:
                    coref_idxs_t.append(
                        [
                            [
                                (t, span[0], span[1])
                                for span in cluster
                                # if t_tokens[0][
                                #     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                                #         t_subtoken_map[0][span[0]]
                                #     ]
                                #     + 1
                                # ][0].lower()
                                # not in PRONOUNS_GROUPS
                                # and span[1] - span[0] < 10
                                # and t_tokens[0][
                                #     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                                #         t_subtoken_map[0][span[0]]
                                #     ]
                                #     + 1
                                # ][0][0].isupper()
                                # and t_tokens[0][
                                #     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                                #         t_subtoken_map[0][span[0]]
                                #     ]
                                #     + 1
                                # ][0][0].isupper()
                            ]
                            for cluster in coreferences
                        ]
                    )
                    # print(
                    #     [
                    #         [
                    #             (
                    #                 t_tokens[0][
                    #                     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                    #                         t_subtoken_map[0][span[0]]
                    #                     ]
                    #                     + 1
                    #                 ][0].lower()
                    #                 not in PRONOUNS_GROUPS,
                    #                 t_tokens[0][
                    #                     t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                    #                         t_subtoken_map[0][span[0]]
                    #                     ]
                    #                     + 1
                    #                 ][0],
                    #             )
                    #             for span in cluster
                    #         ]
                    #         for cluster in coreferences
                    #     ]
                    # )

                    coref_idxs.append(
                        [
                            [(t, span[0], span[1]) for span in cluster]
                            for cluster in coreferences
                            # if len(
                            #     [
                            #         (t, span[0], span[1])
                            #         for span in cluster
                            #         if t_tokens[0][
                            #             t_new_token_map[0][t_subtoken_map[0][span[0]]] : t_new_token_map[0][
                            #                 t_subtoken_map[0][span[0]]
                            #             ]
                            #             + 1
                            #         ][0][0].isupper()
                            #     ]
                            # )
                            # > 0
                        ]
                    )
                    preds["clusters"].append(coreferences)
                    preds["clusters_t"].append([[(t, span[0], span[1]) for span in cluster] for cluster in coreferences])

                for cluster, tttt in zip(coref_idxs[-1], coref_idxs_t[-1]):
                    # if len(tttt) > 0:
                    #     spans = torch.tensor(tttt).to(self.encoder.device)
                    # else:
                    spans = torch.tensor(cluster).to(self.encoder.device)
                    starts_hs = torch.index_select(last_hidden_states, 1, spans[:, 1])
                    ends_hs = torch.index_select(last_hidden_states, 1, spans[:, 2])
                    clusters_ttt = torch.cat((starts_hs, ends_hs), dim=-1)
                    if self.cluster_representation == "soft-dot":
                        cluster_t = self.cluster_transformer(clusters_ttt).repeat(spans.shape[0], 1)
                        span_cluster_t = torch.cat((clusters_ttt.squeeze(0), cluster_t), dim=-1)
                        mention_t_hs.append(span_cluster_t)
                        mention_t_idxs.extend(cluster)
                    elif self.cluster_representation == "soft-mc":
                        mention_t_hs.append(clusters_ttt)
                        cluster_t_hs.append(self.cluster_transformer(clusters_ttt).repeat(spans.shape[0], 1))
                        mention_t_idxs.extend(cluster)

                    if clusters_ttt.shape[1] != 1:
                        if self.cluster_representation == "s2e":
                            start_hs.append(torch.mean(self.tt_coref_representation(starts_hs), dim=1))
                            end_hs.append(torch.mean(self.tt_coref_representation(ends_hs), dim=1))
                        if self.cluster_representation == "dot":
                            coref_hs.append(torch.mean(self.tt_coref_representation(clusters_ttt), dim=1))
                        if self.cluster_representation == "transformer":
                            coref_hs.append(self.cluster_transformer(clusters_ttt))
                    else:
                        if self.cluster_representation == "s2e":
                            start_hs.append(self.t_coref_representation(starts_hs[0]))
                            end_hs.append(self.t_coref_representation(ends_hs[0]))
                        if self.cluster_representation == "dot":
                            coref_hs.append(self.t_coref_representation(clusters_ttt[0]))
                        if self.cluster_representation == "transformer":
                            coref_hs.append(self.cluster_transformer(clusters_ttt))

                loss_dict["coreference_loss"] = coreference_loss
            else:
                coreferences = []
                if stage != "train":
                    preds["clusters"].append([])
            i += 1

        preds["mention_idxs"] = torch.cat(preds["mention_idxs"], dim=0).unsqueeze(0)
        preds["start_idxs"] = torch.cat(preds["start_idxs"], dim=0).unsqueeze(0)

        if self.cluster_representation == "soft-dot":
            if len(mention_t_idxs) != 0:
                coref_hidden_states = torch.cat(mention_t_hs, dim=-2)
                coref_hidden_states = coref_hidden_states.unsqueeze(0)

                temp_loss, fullcoreferences = self.cluster_clustering_s(
                    coref_hidden_states,
                    [[c[0] + c[1], c[0] + c[2]] for c in mention_t_idxs],
                    full_clusters,
                    stage,
                    add,
                    singletons,
                )
                d = {(m[0] + m[1], m[0] + m[2]): m for m in mention_t_idxs}
                loss += temp_loss
                loss_dict["temp_loss"] = temp_loss

                if stage != "train":
                    preds["full_coreferences"] = [[d[(m[0], m[1])] for m in cluster] for cluster in fullcoreferences]
            else:
                preds["full_coreferences"] = []
        if self.cluster_representation == "soft-mc":
            if len(mention_t_idxs) != 0:
                start_hidden_states = torch.cat(mention_t_hs, dim=-2)
                start_hidden_states = start_hidden_states.unsqueeze(0)
                end_hidden_states = torch.cat(cluster_t_hs, dim=-2)
                end_hidden_states = end_hidden_states.unsqueeze(0)

                start_hidden_states = start_hidden_states.squeeze(0)
                coref_hidden_states = [start_hidden_states, end_hidden_states]

                temp_loss, fullcoreferences = self.cluster_clustering_s(
                    coref_hidden_states,
                    [[c[0] + c[1], c[0] + c[2]] for c in mention_t_idxs],
                    full_clusters,
                    stage,
                    add,
                    singletons,
                )
                d = {(m[0] + m[1], m[0] + m[2]): m for m in mention_t_idxs}
                loss += temp_loss
                loss_dict["temp_loss"] = temp_loss

                if stage != "train":
                    [[d[(m[0], m[1])] for m in cluster] for cluster in fullcoreferences]
                    preds["full_coreferences"] = [[d[(m[0], m[1])] for m in cluster] for cluster in fullcoreferences]
            else:
                preds["full_coreferences"] = []
        if len(coref_hs) > 0 or len(start_hs) > 0:
            if self.cluster_representation != "s2e":
                coref_hidden_states = torch.stack(coref_hs).squeeze(1)
                coref_hidden_states = coref_hidden_states.unsqueeze(0)
            else:
                start_hidden_states = torch.stack(start_hs).squeeze(1)
                start_hidden_states = start_hidden_states.unsqueeze(0)
                end_hidden_states = torch.stack(end_hs).squeeze(1)
                end_hidden_states = end_hidden_states.unsqueeze(0)
                coref_hidden_states = [start_hidden_states, end_hidden_states]

            if self.negatives:
                temp_loss, fullcoreferences = self.cluster_clustering3(
                    coref_hidden_states, coref_idxs, full_clusters, stage, add, singletons, ttttttttt
                )
            else:

                temp_loss, fullcoreferences = self.cluster_clustering(
                    coref_hidden_states, coref_idxs, full_clusters, stage, add, singletons
                )

            loss += temp_loss
            loss_dict["temp_loss"] = temp_loss

            if stage != "train":
                preds["full_coreferences"] = fullcoreferences
        else:
            preds["full_coreferences"] = []

        loss_dict["full_loss"] = loss
        outputs = {"pred_dict": preds, "loss_dict": loss_dict, "loss": loss}
        return outputs
