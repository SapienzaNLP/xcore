import json
import hydra
import subprocess
import torch
from omegaconf import omegaconf
from xcore.common.util import *
from xcore.common.metrics import *
from tqdm import tqdm
from data.pl_data_modules import CrossDataModule
from models.pl_modules import CrossPLModule
from xcore.utils.loggingl import get_console_logger

logger = get_console_logger()


def jsonlines_to_html(jsonlines_input_name, output):
    cwd = str(hydra.utils.get_original_cwd())
    subprocess.call(
        "python3 "
        + cwd
        + "/xcore/utils/corefconversion/jsonlines2text.py "
        + cwd
        + "/"
        + jsonlines_input_name
        + " -i  -o "
        + cwd
        + "/experiments/"
        + output
        + ".html --sing-color"
        ' "black" --cm "common"',
        shell=True,
    )


@torch.no_grad()
def evaluate(conf: omegaconf.DictConfig):
    device = conf.evaluation.device
    
    hydra.utils.log.info("Using {} as device".format(device))
    pl_data_module: CrossDataModule = hydra.utils.instantiate(conf.data.datamodule, _recursive_=False)

    pl_data_module.prepare_data()
    pl_data_module.setup("test")
    jsonlines_to_html(pl_data_module.test_dataloader().dataset.path, "test")

    logger.log(f"Instantiating the Model from {conf.evaluation.checkpoint}")
    model = CrossPLModule.load_from_checkpoint(conf.evaluation.checkpoint, _recursive_=False, map_location=device)
    gold = []
    info = []
    with open(hydra.utils.get_original_cwd() + "/" + pl_data_module.test_dataloader().dataset.path, "r") as f:
        for line in f.readlines():
            doc = json.loads(line)
            if "sentences" in doc:
                info.append({"doc_key": doc["doc_key"], "sentences": doc["sentences"]})
            clusters = []
            if "clusters" in doc:
                for cluster in doc["clusters"]:
                    if not conf.evaluation.singletons and len(cluster) < 2:
                        continue
                    clusters.append(tuple([(m[0], m[1]) for m in cluster]))
            gold.append(clusters)
        mention_to_gold_clusters = [extract_mentions_to_clusters([tuple(g) for g in gold_element]) for gold_element in gold]

        device_and_singletons ={"device": device,
                                "singletons": conf.evaluation.singletons}
        predictions = model_predictions_with_dataloader(model, pl_data_module.test_dataloader(), device_and_singletons)
        mention_to_predicted_clusters = [extract_mentions_to_clusters(p) for p in predictions]

        print(evaluate_coref_scores(predictions, gold, mention_to_predicted_clusters, mention_to_gold_clusters))

        with open(hydra.utils.get_original_cwd() + "/experiments/output.jsonlines", "w") as f:
            for pred, infos in zip(predictions, info):
                infos["clusters"] = pred
                f.write(json.dumps(infos) + "\n")

        jsonlines_to_html("experiments/output.jsonlines", "output")
    return


def evaluate_coref_scores(pred, gold, mention_to_pred, mention_to_gold):
    evaluator = OfficialCoNLL2012CorefEvaluator()

    for p, g, m2p, m2g in zip(pred, gold, mention_to_pred, mention_to_gold):
        evaluator.update(p, g, m2p, m2g)
    result = []
    for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
        result.append(dict(zip(["precision", "recall", "f1_score"], evaluator.get_prf(metric))))
    return result


def model_predictions_with_dataloader(model, test_dataloader, device_and_singletons):
    model.to(device_and_singletons["device"])
    model.eval()
    predictions = []

    for batch in tqdm(test_dataloader, desc="Test", total=test_dataloader.__len__()):
        output = model.model(
            stage="temp",
            input_ids=[elem.to(device_and_singletons["device"]) for elem in batch["index_input_ids"]],
            attention_mask=[elem.to(device_and_singletons["device"]) for elem in batch["index_attention_mask"]],
            eos_mask=[elem.to(device_and_singletons["device"]) for elem in batch["index_eos_mask"]],
            gold_starts=[elem.to(device_and_singletons["device"]) for elem in batch["index_gold_starts"]],
            gold_mentions=[elem.to(device_and_singletons["device"]) for elem in batch["index_gold_mentions"]],
            gold_clusters=batch["index_gold_clusters"],
            singletons=device_and_singletons["singletons"],
            full_clusters=batch["gold_c"].to(device_and_singletons["device"]),
            temp=batch["temp"],
            tokens=batch["t_tokens"],
            subtoken_map=batch["t_subtoken_map"],
            new_token_map=batch["t_new_token_map"],
        )

        clusters_predicted = original_token_offsets3(
            clusters=output["pred_dict"]["full_coreferences"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0],
        )
        predictions.append(clusters_predicted)

    return predictions


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    evaluate(conf)


if __name__ == "__main__":
    main()
