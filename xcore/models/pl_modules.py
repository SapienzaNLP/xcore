from typing import Any

import hydra
import pytorch_lightning as pl
import torch

from torchmetrics import *
import transformers
from transformers import Adafactor

from xcore.common.metrics import *
from xcore.models.model_cross import xCoRe_system
from xcore.common.util import *



class CrossPLModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        try:
            self.model = hydra.utils.instantiate(self.hparams.model)
        except:
            self.hparams.model["_target_"] = "xcore." + self.hparams.model["_target_"]
            self.model = hydra.utils.instantiate(self.hparams.model)
        self.train_step_predictions = []
        self.train_step_gold = []
        self.validation_step_predictions = []
        self.validation_step_gold = []
        self.test_step_predictions = []
        self.test_step_gold = []

    def forward(self, batch) -> dict:
        output_dict = self.model(batch, "forward")
        return output_dict

    def evaluate(self, predictions, golds):
        mention_evaluator = OfficialMentionEvaluator()
        start_evaluator = OfficialMentionEvaluator()
        cluster_mention_evaluator = OfficialMentionEvaluator()
        coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        full_coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        t_coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        result = {}

        for pred, gold in zip(predictions, golds):
            if "start_idxs" in pred.keys() and "gold_starts" in gold.keys():
                starts_pred = pred["start_idxs"][0].tolist()
                starts_gold = (gold["gold_starts"][0] == 1).nonzero(as_tuple=False).squeeze(-1).tolist()
                start_evaluator.update(starts_pred, starts_gold)

            if "mention_idxs" in pred.keys() and "gold_mentions" in gold.keys():
                mentions_pred = [tuple(p) for p in pred["mention_idxs"][0].tolist()]
                mentions_gold = [tuple(g) for g in (gold["gold_mentions"][0] == 1).nonzero(as_tuple=False).tolist()]
                mention_evaluator.update(mentions_pred, mentions_gold)

            if "clusters" in pred.keys():
                for i, clusters in enumerate(pred["clusters"]):

                    pred_clusters = clusters
                    gold_clusters = gold["index_gold_clusters"][i]
                    mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                    mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                    coref_evaluator.update(
                        pred_clusters,
                        gold_clusters,
                        mention_to_predicted_clusters,
                        mention_to_gold_clusters,
                    )

                    cluster_mention_evaluator.update(
                        [item for sublist in gold_clusters for item in sublist],
                        [item for sublist in pred_clusters for item in sublist],
                    )

            if "full_coreferences" in pred.keys():
                pred_clusters = pred["full_coreferences"]
                gold_clusters = gold["gold_clusters"]
                mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                full_coref_evaluator.update(
                    pred_clusters,
                    gold_clusters,
                    mention_to_predicted_clusters,
                    mention_to_gold_clusters,
                )

            if "full_coreferences_t" in pred.keys():
                pred_clusters = pred["full_coreferences_t"]
                gold_clusters = gold["gold_clusters"]
                mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                t_coref_evaluator.update(
                    pred_clusters,
                    gold_clusters,
                    mention_to_predicted_clusters,
                    mention_to_gold_clusters,
                )

        p, r, f1 = start_evaluator.get_prf()
        result.update(
            {
                "start_f1_score": f1,
                "start_precision": p,
                "start_recall": r,
            }
        )
        p, r, f1 = mention_evaluator.get_prf()
        result.update({"mention_f1_score": f1, "mention_precision": p, "mention_recall": r})
        p, r, f1 = cluster_mention_evaluator.get_prf()
        result.update(
            {
                "cluster_mention_f1_score": f1,
                "cluster_mention_precision": p,
                "cluster_mention_recall": r,
            }
        )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = coref_evaluator.get_prf(metric)
            result.update(
                {
                    metric + "_f1_score": f1,
                    metric + "_precision": p,
                    metric + "_recall": r,
                }
            )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = full_coref_evaluator.get_prf(metric)
            result.update(
                {
                    "full_" + metric + "_f1_score": f1,
                    "full_" + metric + "_precision": p,
                    "full_" + metric + "_recall": r,
                }
            )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = t_coref_evaluator.get_prf(metric)
            result.update(
                {
                    "full_t_" + metric + "_f1_score": f1,
                    "full_t_" + metric + "_precision": p,
                    "full_t_" + metric + "_recall": r,
                }
            )
        return result

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self.model(
            stage="train",
            input_ids=batch["index_input_ids"],
            attention_mask=batch["index_attention_mask"],
            eos_mask=batch["index_eos_mask"],
            gold_starts=batch["index_gold_starts"],
            gold_mentions=batch["index_gold_mentions"],
            gold_clusters=batch["index_gold_clusters"],
            singletons=batch["singletons"],
            full_clusters=batch["gold_c"],
            temp=batch["temp"],
            tokens=batch["t_tokens"],
            subtoken_map=batch["t_subtoken_map"],
            new_token_map=batch["t_new_token_map"],
        )

        self.log_dict({"train/" + k: v for k, v in output["loss_dict"].items()}, on_step=True)
        return output["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        output = self.model(
            stage="temp",
            input_ids=batch["index_input_ids"],
            attention_mask=batch["index_attention_mask"],
            eos_mask=batch["index_eos_mask"],
            gold_starts=batch["index_gold_starts"],
            gold_mentions=batch["index_gold_mentions"],
            gold_clusters=batch["index_gold_clusters"],
            singletons=batch["singletons"],
            full_clusters=batch["gold_c"],
            temp=batch["temp"],
            tokens=batch["t_tokens"],
            subtoken_map=batch["t_subtoken_map"],
            new_token_map=batch["t_new_token_map"],
        )
        self.log_dict({"val/" + k: v for k, v in output["loss_dict"].items()})
        # output["pred_dict"]["full_coreferences"] = [
        #     ((s[0] + s[1], s[0] + s[2]) for s in cluster) for cluster in output["pred_dict"]["full_coreferences"]
        # ]

        output["pred_dict"]["full_coreferences_t"] = original_token_offsets3(
            clusters=[x for xx in output["pred_dict"]["clusters_t"] for x in xx],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )

        output["pred_dict"]["full_coreferences"] = original_token_offsets3(
            clusters=output["pred_dict"]["full_coreferences"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )

        self.validation_step_predictions.append(output["pred_dict"])

        gold = {
            "index_gold_clusters": [
                [tuple([(i[0], i[1]) for i in cluster]) for cluster in temppppppp] for temppppppp in batch["tempppp"]
            ],
            "gold_clusters": original_token_offsets(
                clusters=unpad_gold_clusters(batch["gold_clusters"].cpu()),
                subtoken_map=batch["subtoken_map"][0],
                new_token_map=batch["new_token_map"][0],
            ),
        }
        if "gold_starts" in batch:
            gold["gold_starts"] = batch["gold_starts"].cpu()
            gold["gold_mentions"] = batch["gold_mentions"].cpu()

        self.validation_step_gold.append(gold)

    def on_validation_epoch_end(self):
        self.log_dict(
            {"val/" + k: v for k, v in self.evaluate(self.validation_step_predictions, self.validation_step_gold).items()}
        )
        self.validation_step_predictions = []
        self.validation_step_gold = []

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        output = self.model(
            stage="test",
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_mask=batch["eos_mask"],
            tokens=batch["tokens"],
            subtoken_map=batch["subtoken_map"],
            new_token_map=batch["new_token_map"],
            singletons=batch["singletons"],
        )
        self.log_dict({"test/" + k: v for k, v in output["loss_dict"].items()})
        output["pred_dict"]["clusters"] = original_token_offsets(
            clusters=output["pred_dict"]["clusters"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )
        self.test_step_predictions.append(output["pred_dict"])
        self.test_step_gold.append(
            {
                "gold_starts": batch["gold_starts"].cpu(),
                "gold_mentions": batch["gold_mentions"].cpu(),
                "gold_clusters": original_token_offsets(
                    clusters=unpad_gold_clusters(batch["gold_clusters"].cpu()),
                    subtoken_map=batch["subtoken_map"][0],
                    new_token_map=batch["new_token_map"][0],
                ),
            }
        )

    def on_test_epoch_end(self):
        self.log_dict({"test/" + k: v for k, v in self.evaluate(self.test_step_predictions, self.test_step_gold).items()})
        self.test_step_predictions = []
        self.test_step_gold = []

    def configure_optimizers(self):
        if self.hparams.opt == "RAdam":
            opt = hydra.utils.instantiate(self.hparams.RAdam, params=self.parameters())
            return opt
        else:
            return self.custom_opt()

    def custom_opt(self):
        no_decay = ["bias", "LayerNorm.weight"]
        head_params = ["representaion", "classifier"]

        model_decay = [
            p
            for n, p in self.model.named_parameters()
            if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        model_no_decay = [
            p
            for n, p in self.model.named_parameters()
            if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]
        head_decay = [
            p
            for n, p in self.model.named_parameters()
            if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        head_no_decay = [
            p for n, p in self.model.named_parameters() if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]

        head_learning_rate = 3e-4
        lr = 2e-5
        wd = 0.01
        optimizer_grouped_parameters = [
            {"params": model_decay, "lr": lr, "weight_decay": wd},
            {"params": model_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_learning_rate, "weight_decay": wd},
            {"params": head_no_decay, "lr": head_learning_rate, "weight_decay": 0.0},
        ]
        optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
        # optimizer = torch.optim.RAdam(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.lr_scheduler.num_training_steps * 0.1,
            num_training_steps=self.hparams.lr_scheduler.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

#remove when done
class s7_tempPLModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model["_target_"] = 'xcore.models.model_cross.Maverick_cross'
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.train_step_predictions = []
        self.train_step_gold = []
        self.validation_step_predictions = []
        self.validation_step_gold = []
        self.test_step_predictions = []
        self.test_step_gold = []

    def forward(self, batch) -> dict:
        output_dict = self.model(batch, "forward")
        return output_dict

    def evaluate(self, predictions, golds):
        mention_evaluator = OfficialMentionEvaluator()
        start_evaluator = OfficialMentionEvaluator()
        cluster_mention_evaluator = OfficialMentionEvaluator()
        coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        full_coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        t_coref_evaluator = OfficialCoNLL2012CorefEvaluator()
        result = {}

        for pred, gold in zip(predictions, golds):
            if "start_idxs" in pred.keys() and "gold_starts" in gold.keys():
                starts_pred = pred["start_idxs"][0].tolist()
                starts_gold = (gold["gold_starts"][0] == 1).nonzero(as_tuple=False).squeeze(-1).tolist()
                start_evaluator.update(starts_pred, starts_gold)

            if "mention_idxs" in pred.keys() and "gold_mentions" in gold.keys():
                mentions_pred = [tuple(p) for p in pred["mention_idxs"][0].tolist()]
                mentions_gold = [tuple(g) for g in (gold["gold_mentions"][0] == 1).nonzero(as_tuple=False).tolist()]
                mention_evaluator.update(mentions_pred, mentions_gold)

            if "clusters" in pred.keys():
                for i, clusters in enumerate(pred["clusters"]):

                    pred_clusters = clusters
                    gold_clusters = gold["index_gold_clusters"][i]
                    mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                    mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                    coref_evaluator.update(
                        pred_clusters,
                        gold_clusters,
                        mention_to_predicted_clusters,
                        mention_to_gold_clusters,
                    )

                    cluster_mention_evaluator.update(
                        [item for sublist in gold_clusters for item in sublist],
                        [item for sublist in pred_clusters for item in sublist],
                    )

            if "full_coreferences" in pred.keys():
                pred_clusters = pred["full_coreferences"]
                gold_clusters = gold["gold_clusters"]
                mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                full_coref_evaluator.update(
                    pred_clusters,
                    gold_clusters,
                    mention_to_predicted_clusters,
                    mention_to_gold_clusters,
                )

            if "full_coreferences_t" in pred.keys():
                pred_clusters = pred["full_coreferences_t"]
                gold_clusters = gold["gold_clusters"]
                mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
                mention_to_predicted_clusters = extract_mentions_to_clusters(pred_clusters)

                t_coref_evaluator.update(
                    pred_clusters,
                    gold_clusters,
                    mention_to_predicted_clusters,
                    mention_to_gold_clusters,
                )

        p, r, f1 = start_evaluator.get_prf()
        result.update(
            {
                "start_f1_score": f1,
                "start_precision": p,
                "start_recall": r,
            }
        )
        p, r, f1 = mention_evaluator.get_prf()
        result.update({"mention_f1_score": f1, "mention_precision": p, "mention_recall": r})
        p, r, f1 = cluster_mention_evaluator.get_prf()
        result.update(
            {
                "cluster_mention_f1_score": f1,
                "cluster_mention_precision": p,
                "cluster_mention_recall": r,
            }
        )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = coref_evaluator.get_prf(metric)
            result.update(
                {
                    metric + "_f1_score": f1,
                    metric + "_precision": p,
                    metric + "_recall": r,
                }
            )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = full_coref_evaluator.get_prf(metric)
            result.update(
                {
                    "full_" + metric + "_f1_score": f1,
                    "full_" + metric + "_precision": p,
                    "full_" + metric + "_recall": r,
                }
            )

        for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
            p, r, f1 = t_coref_evaluator.get_prf(metric)
            result.update(
                {
                    "full_t_" + metric + "_f1_score": f1,
                    "full_t_" + metric + "_precision": p,
                    "full_t_" + metric + "_recall": r,
                }
            )
        return result

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self.model(
            stage="train",
            input_ids=batch["index_input_ids"],
            attention_mask=batch["index_attention_mask"],
            eos_mask=batch["index_eos_mask"],
            gold_starts=batch["index_gold_starts"],
            gold_mentions=batch["index_gold_mentions"],
            gold_clusters=batch["index_gold_clusters"],
            singletons=batch["singletons"],
            full_clusters=batch["gold_c"],
            temp=batch["slices_seq_index"],
            tokens=batch["t_tokens"],
            subtoken_map=batch["t_subtoken_map"],
            new_token_map=batch["t_new_token_map"],
        )

        self.log_dict({"train/" + k: v for k, v in output["loss_dict"].items()}, on_step=True)
        return output["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        output = self.model(
            stage="temp",
            input_ids=batch["index_input_ids"],
            attention_mask=batch["index_attention_mask"],
            eos_mask=batch["index_eos_mask"],
            gold_starts=batch["index_gold_starts"],
            gold_mentions=batch["index_gold_mentions"],
            gold_clusters=batch["index_gold_clusters"],
            singletons=batch["singletons"],
            full_clusters=batch["gold_c"],
            temp=batch["slices_seq_index"],
            tokens=batch["t_tokens"],
            subtoken_map=batch["t_subtoken_map"],
            new_token_map=batch["t_new_token_map"],
        )
        self.log_dict({"val/" + k: v for k, v in output["loss_dict"].items()})
        # output["pred_dict"]["full_coreferences"] = [
        #     ((s[0] + s[1], s[0] + s[2]) for s in cluster) for cluster in output["pred_dict"]["full_coreferences"]
        # ]

        output["pred_dict"]["full_coreferences_t"] = original_token_offsets3(
            clusters=[x for xx in output["pred_dict"]["clusters_t"] for x in xx],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )

        output["pred_dict"]["full_coreferences"] = original_token_offsets3(
            clusters=output["pred_dict"]["full_coreferences"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )

        self.validation_step_predictions.append(output["pred_dict"])

        ttt = {
            "index_gold_clusters": [
                [tuple([(i[0], i[1]) for i in cluster]) for cluster in temppppppp] for temppppppp in batch["tempppp"]
            ],
            "gold_clusters": original_token_offsets(
                clusters=unpad_gold_clusters(batch["gold_clusters"].cpu()),
                subtoken_map=batch["subtoken_map"][0],
                new_token_map=batch["new_token_map"][0],
            ),
        }
        if "gold_starts" in batch:
            ttt["gold_starts"] = batch["gold_starts"].cpu()
            ttt["gold_mentions"] = batch["gold_mentions"].cpu()

        self.validation_step_gold.append(ttt)

    def on_validation_epoch_end(self):
        self.log_dict(
            {"val/" + k: v for k, v in self.evaluate(self.validation_step_predictions, self.validation_step_gold).items()}
        )
        self.validation_step_predictions = []
        self.validation_step_gold = []

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        output = self.model(
            stage="test",
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_mask=batch["eos_mask"],
            tokens=batch["tokens"],
            subtoken_map=batch["subtoken_map"],
            new_token_map=batch["new_token_map"],
            singletons=batch["singletons"],
        )
        self.log_dict({"test/" + k: v for k, v in output["loss_dict"].items()})
        output["pred_dict"]["clusters"] = original_token_offsets(
            clusters=output["pred_dict"]["clusters"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )
        self.test_step_predictions.append(output["pred_dict"])
        self.test_step_gold.append(
            {
                "gold_starts": batch["gold_starts"].cpu(),
                "gold_mentions": batch["gold_mentions"].cpu(),
                "gold_clusters": original_token_offsets(
                    clusters=unpad_gold_clusters(batch["gold_clusters"].cpu()),
                    subtoken_map=batch["subtoken_map"][0],
                    new_token_map=batch["new_token_map"][0],
                ),
            }
        )

    def on_test_epoch_end(self):
        self.log_dict({"test/" + k: v for k, v in self.evaluate(self.test_step_predictions, self.test_step_gold).items()})
        self.test_step_predictions = []
        self.test_step_gold = []

    def configure_optimizers(self):
        if self.hparams.opt == "RAdam":
            opt = hydra.utils.instantiate(self.hparams.RAdam, params=self.parameters())
            return opt
        else:
            return self.custom_opt()

    def custom_opt(self):
        no_decay = ["bias", "LayerNorm.weight"]
        head_params = ["representaion", "classifier"]

        model_decay = [
            p
            for n, p in self.model.named_parameters()
            if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        model_no_decay = [
            p
            for n, p in self.model.named_parameters()
            if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]
        head_decay = [
            p
            for n, p in self.model.named_parameters()
            if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        head_no_decay = [
            p for n, p in self.model.named_parameters() if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]

        head_learning_rate = 3e-4
        lr = 2e-5
        wd = 0.01
        optimizer_grouped_parameters = [
            {"params": model_decay, "lr": lr, "weight_decay": wd},
            {"params": model_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_learning_rate, "weight_decay": wd},
            {"params": head_no_decay, "lr": head_learning_rate, "weight_decay": 0.0},
        ]
        optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
        # optimizer = torch.optim.RAdam(optimizer_grouped_parameters, scale_parameter=False, relative_step=False)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.lr_scheduler.num_training_steps * 0.1,
            num_training_steps=self.hparams.lr_scheduler.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
