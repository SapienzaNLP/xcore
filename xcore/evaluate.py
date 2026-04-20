import os
import re
import json
import operator
import subprocess
import collections
from pprint import pprint
import hydra
import torch
from tqdm import tqdm
from omegaconf import omegaconf
from models.pl_modules import CrossPLModule
from data.pl_data_modules import CrossDataModule
from xcore.common.util import (extract_mentions_to_clusters,
        original_token_offsets3)
from xcore.common.metrics import OfficialCoNLL2012CorefEvaluator
from xcore.utils.loggingl import get_console_logger

logger = get_console_logger()
# extract doc_key and part given format 'doc_key.p.0'
DOC_KEY_RE = re.compile(r'^(.+)\.p\.(\d+)$')


def jsonlines_to_html(jsonlines_input_name, output):
    cwd = str(hydra.utils.get_original_cwd())
    subprocess.check_call(
            ["python3",
            cwd + "/xcore/utils/corefconversion/jsonlines2text.py",
            jsonlines_input_name,
            "-i",
            "-o", output,
            "--sing-color", "black",
            "--cm", "common"],
            shell=False)


def output_conll(info, output_file):
    # Based on https://github.com/kentonl/e2e-coref/blob/master/conll.py
    with open(output_file, 'w', encoding='utf8') as out:
        for infos in info:
            doc_key = infos['doc_key']
            clusters = infos['clusters']
            start_map = collections.defaultdict(list)
            end_map = collections.defaultdict(list)
            word_map = collections.defaultdict(list)
            for cluster_id, mentions in enumerate(clusters):
                for start, end in mentions:
                    if start == end:
                        word_map[start].append(cluster_id)
                    else:
                        start_map[start].append((cluster_id, end))
                        end_map[end].append((cluster_id, start))
            for k,v in start_map.items():
                start_map[k] = [cluster_id for cluster_id, end
                        in sorted(v, key=operator.itemgetter(1), reverse=True)]
            for k,v in end_map.items():
                end_map[k] = [cluster_id for cluster_id, start
                        in sorted(v, key=operator.itemgetter(1), reverse=True)]

            docname, part = DOC_KEY_RE.match(doc_key).groups()
            print(f'#begin document ({docname}); part {part}', file=out)
            word_index = 0
            for sentence in infos['sentences']:
                for tokenid, token in enumerate(sentence):
                    row = [docname, part, tokenid, token] + ['-'] * 8
                    coref_list = []
                    if word_index in end_map:
                        for cluster_id in end_map[word_index]:
                            coref_list.append(f'{cluster_id})')
                    if word_index in word_map:
                        for cluster_id in word_map[word_index]:
                            coref_list.append(f'({cluster_id})')
                    if word_index in start_map:
                        for cluster_id in start_map[word_index]:
                            coref_list.append(f'({cluster_id}')
                    row[-1] = '|'.join(coref_list) if coref_list else '-'
                    print(*row, sep='\t', file=out)
                    word_index += 1
                print('', file=out)
            print('#end document', file=out)


@torch.no_grad()
def evaluate(conf: omegaconf.DictConfig):
    device = conf.evaluation.device

    hydra.utils.log.info('Using %s as device', device)
    pl_data_module: CrossDataModule = hydra.utils.instantiate(
            conf.data.datamodule, _recursive_=False)

    pl_data_module.prepare_data()
    pl_data_module.setup("test")
    cwd = str(hydra.utils.get_original_cwd())
    # 'data/mydataset/dev.jsonlines' -> 'mydataset'
    dataset = os.path.basename(os.path.dirname(
            pl_data_module.test_dataloader().dataset.path))
    # Write all output to a sibling directory of the checkpoints directory
    # experiments/xcore/myexperiment/wandb/run-2026{...}/files/{dataset}
    # This means we can evaluate a single checkpoint on multiple test datasets
    # and keep the results in separate directories.
    outpath = os.path.dirname(conf.evaluation.checkpoint
            ).removesuffix('checkpoints') + dataset
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    # All output filenames are of the form {subset}_{modality},
    # where modality is gold or output.
    jsonlines_to_html(
            cwd + '/' + pl_data_module.test_dataloader().dataset.path,
            outpath + "/test_gold.html")
    logger.log(f"Instantiating the Model from {conf.evaluation.checkpoint}")
    model = CrossPLModule.load_from_checkpoint(
            conf.evaluation.checkpoint, _recursive_=False,
            map_location=device, weights_only=False)

    gold = []
    info = []
    with open(cwd + "/" + pl_data_module.test_dataloader().dataset.path,
            'r', encoding='utf8') as infile:
        for line in infile.readlines():
            doc = json.loads(line)
            if "sentences" in doc:
                info.append({
                        "doc_key": doc["doc_key"],
                        "sentences": doc["sentences"]})
            clusters = []
            if "clusters" in doc:
                for cluster in doc["clusters"]:
                    if not conf.evaluation.singletons and len(cluster) < 2:
                        continue
                    clusters.append(tuple((m[0], m[1]) for m in cluster))
            gold.append(clusters)
    mention_to_gold_clusters = [
            extract_mentions_to_clusters([tuple(g) for g in gold_element])
            for gold_element in gold]

    predictions = model_predictions_with_dataloader(
            model, pl_data_module.test_dataloader(), device,
            conf.evaluation.singletons)
    mention_to_predicted_clusters = [extract_mentions_to_clusters(p)
            for p in predictions]

    pprint(evaluate_coref_scores(predictions, gold,
                mention_to_predicted_clusters, mention_to_gold_clusters),
            sort_dicts=False)

    with open(outpath + '/test_output.jsonlines', 'w',
            encoding='utf8') as outfile:
        for pred, infos in zip(predictions, info):
            infos["clusters"] = pred
            outfile.write(json.dumps(infos) + "\n")

    jsonlines_to_html(
            outpath + "/test_output.jsonlines",
            outpath + "/test_output.html")

    output_conll(info, outpath + '/test_output.conll')


def evaluate_coref_scores(pred, gold, mention_to_pred, mention_to_gold):
    evaluator = OfficialCoNLL2012CorefEvaluator()

    for p, g, m2p, m2g in zip(pred, gold, mention_to_pred, mention_to_gold):
        evaluator.update(p, g, m2p, m2g)
    result = {}
    for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
        result[metric] = dict(zip([
                "precision", "recall", "f1_score"],
                evaluator.get_prf(metric)))
    return result


def model_predictions_with_dataloader(
        model, test_dataloader, device, singletons):
    model.to(device)
    model.eval()
    predictions = []

    for batch in tqdm(
            test_dataloader, desc="Test", total=len(test_dataloader)):
        output = model.model(
            stage="temp",
            input_ids=[elem.to(device) for elem in batch["index_input_ids"]],
            attention_mask=[elem.to(device)
                    for elem in batch["index_attention_mask"]],
            eos_mask=[elem.to(device) for elem in batch["index_eos_mask"]],
            gold_starts=[elem.to(device)
                    for elem in batch["index_gold_starts"]],
            gold_mentions=[elem.to(device)
                    for elem in batch["index_gold_mentions"]],
            gold_clusters=batch["index_gold_clusters"],
            singletons=singletons,
            full_clusters=batch["gold_c"].to(device),
            temp=batch["temp"],
            tokens=batch["t_tokens"],
            subtoken_map=batch["t_subtoken_map"],
            new_token_map=batch["t_new_token_map"])

        clusters_predicted = original_token_offsets3(
            clusters=output["pred_dict"]["full_coreferences"],
            subtoken_map=batch["subtoken_map"][0],
            new_token_map=batch["new_token_map"][0])
        predictions.append(clusters_predicted)

    return predictions


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    evaluate(conf)


if __name__ == '__main__':
    main()
