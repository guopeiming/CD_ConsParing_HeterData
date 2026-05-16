from cross_domain_constituency_parsing.utils.tree_structure import parse_bracketed_parse_tree
from cross_domain_constituency_parsing import BaseConstituencyParserPredictor
from allennlp.models.archival import load_archive
import jsonlines
import re
import os
from tqdm import tqdm
from random import shuffle
from typing import List, Tuple
import copy
from transformers import BertTokenizer


def build_base_cp_dataset():
    input_path: str = "./data/origin/WSJ/test.txt"
    output_path: str = "./data/base/consti_parsing/test.jsonl"
    with open(input_path, "r", encoding="utf-8") as reader, jsonlines.open(output_path, "w") as writer:
        for line in tqdm(reader):
            line = line.strip()
            tree = parse_bracketed_parse_tree(line)
            writer.write({
                "tokens": list(tree.leaves()),
                "task": "constituency_parsing",
                "domain": "news",
                "language": "en",
                "linearized_tree": line
            })


def build_joint_lm_cp_dataset(cp_path: str, lm_path: str, output_path: str, num_lm: int, domain: str):
    lm_sents: List[str] = []

    with jsonlines.open(output_path, "w") as writer:
        with jsonlines.open(cp_path, "r") as reader:
            for line in tqdm(reader):
                writer.write(line)

        with open(lm_path, "r", encoding="utf-8") as reader:
            for line in tqdm(reader):
                line = line.strip()
                lm_sents.append(line)

        shuffle(lm_sents)
        assert num_lm <= len(lm_sents), "num_lm error"
        lm_sents = lm_sents[:num_lm]

        for sent in tqdm(lm_sents):
            tree = parse_bracketed_parse_tree(sent)
            tokens = list(tree.leaves())
            writer.write({
                "tokens": tokens,
                "task": "language_model",
                "domain": domain,
                "language": "en",
                "linearized_tree": sent
            })


def token_filter(token: str) -> List[str]:
    if token == "":
        return []

    if token.startswith("("):
        return ["-LRB-"] + token_filter(token[1:])

    if token.endswith(")"):
        return  token_filter(token[:-1]) + ["-RRB-"]

    if token.startswith("["):
        return ["-LSB-"] + token_filter(token[1:])

    if token.endswith("]"):
        return token_filter(token[:-1]) + ["-RSB-"]

    if token.startswith("{"):
        return ["-LCB-"] + token_filter(token[1:])

    if token.endswith("}"):
        return token_filter(token[:-1]) + ["-RCB-"]

    if token.endswith(("'m", "'s")):
        return token_filter(token[:-2]) + [token[-2:]]

    if token.endswith(("n't", "'ve", "'re")):
        return token_filter(token[:-3]) + [token[-3:]]

    if token.startswith((".", ",", "?", "!", "\"", "`", "\'", ":", ";", "$", "<", "-", "_", "%")):
        split_index = 1
        for i in range(1, len(token)):
            if token[i] != token[0]:
                break
            else:
                split_index += 1
        return [token[:split_index]] + token_filter(token[split_index:])

    if token.endswith((".", ",", "?", "!", "\"", "`", "\'", ":", ";", "$", ">", "-", "_", "%")):
        split_index = -1
        for i in range(-2, -len(token)-1, -1):
            if token[i] != token[-1]:
                break
            else:
                split_index -= 1
        return token_filter(token[0:split_index]) + [token[split_index:]]

    return [token]


def raw_corpus_generator(filepath: str, batch_size: int = 50) -> List[List[str]]:
    batch_sents = []
    tokenizer = BertTokenizer.from_pretrained("data/plms/bert-large-uncased/")

    with open(filepath, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line == "":
                continue
            line = line + " "

            sents = re.findall(r".*?[\!\?\.] ", line)
            for sent in sents:
                tokens = []
                for token in sent.split():
                    tokens.extend(token_filter(token))

                sent = " ".join(tokens)
                if (len(tokens) >= 140) or (len(tokens) <= 10) or (re.search(r"[\(\)\{\}\[\]]", sent) is not None) or len(tokenizer(sent)["input_ids"]) > 150:
                    continue
                batch_sents.append(tokens)

                if len(batch_sents) >= batch_size:
                    yield batch_sents
                    batch_sents = []

        if len(batch_sents) > 0:
            yield batch_sents


def clean_and_parse_raw_corpus(input_file: str, output_file: str, model_dir: str, cuda_device: int = 0) -> None:
    num = 0
    archive = load_archive(model_dir, cuda_device=cuda_device)
    parser = BaseConstituencyParserPredictor(archive.model, archive.validation_dataset_reader)

    with open(output_file, "w", encoding="utf-8") as writer:
        for sents in tqdm(raw_corpus_generator(input_file)):
            trees = parser.predict([{"tokens": sent, "postags": ["PAD_TAG"]*len(sent)} for sent in sents])
            trees = [tree["pred_linearized_tree"] for tree in trees]

            writer.write("\n".join(trees)+"\n")
            num += len(sents)

            if num % 10000 == 0:
                print(f"{num} sentences were processed.")
            if num >= 80000:
                break

    print(f"{num} sentence were written to {output_file}")


def restaurant_tag_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    spans, prev_bio_tag = [], None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == "b":
            spans.append([idx, idx, label])
        elif bio_tag == "i":
            assert prev_bio_tag in ("i", "b") and len(spans) > 0 and label == spans[-1][2], "error"
            spans[-1][1] = idx
        elif bio_tag == "o":
            pass
        else:
            raise ValueError()
        prev_bio_tag = bio_tag

    return spans


def mit_restaurant_preprocess(parser) -> None:
    filedirs: str = [
        "data/origin/mit_restaurant/restauranttrain.bio.txt",
        "data/origin/mit_restaurant/restauranttest.bio.txt"
    ]
    with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "w") as writer:
        for sents, sents_labels in tqdm(ner_corpus_generator(filedirs, 50, 5, 1, 0, restaurant_tag_to_spans)):
            trees = parser.predict([{"tokens": sent, "postags": ["PAD_TAG"]*len(sent)} for sent in sents])
            trees = [tree["pred_linearized_tree"] for tree in trees]

            for i in range(len(sents)):
                writer.write({
                    "tokens": sents[i],
                    "task": "named_entity_recognition",
                    "domain": "restaurant",
                    "language": "en",
                    "linearized_tree": trees[i],
                    "ner_label": sents_labels[i],
                })


def conll03_tag_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    spans, prev_label_tag = [], None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == "i":
            if prev_label_tag == label:
                spans[-1][1] = idx
            else:
                spans.append([idx, idx, label])
        elif bio_tag == "b":
            spans.append([idx, idx, label])
        elif bio_tag == "o":
            pass
        else:
            raise ValueError()
        prev_label_tag = label

    return spans


def ner_corpus_generator(files: List[str], batch_size: int, filter_len: int, token_index: int, label_index: int, fun_tag2span):
    snts, snts_labels = [], []
    tokens, labels = [], []

    for filedir in files:
        with open(filedir, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if line == "":
                    if len(tokens) >= filter_len:
                        snts.append(tokens)
                        snts_labels.append(fun_tag2span(labels))
                    tokens, labels = [], []

                    if len(snts) >= batch_size:
                        yield snts, snts_labels
                        snts, snts_labels = [], []
                else:
                    line = line.split()
                    if line[token_index] == "(":
                        line[token_index] = "-LRB-"
                    if line[token_index] == ")":
                        line[token_index] = "-RRB-"
                    if "(" in line[token_index]:
                        print(line[token_index])
                        line[token_index] = line[token_index].replace("(", "")
                    if ")" in line[token_index]:
                        print(line[token_index])
                        line[token_index] = line[token_index].replace(")", "")
                    tokens.append(line[token_index])
                    labels.append(line[label_index])

    if len(snts) > 0:
        yield snts, snts_labels


def conll03_preprocess(parser: BaseConstituencyParserPredictor) -> None:
    filedirs: str = [
        "data/origin/conll03/eng.train",
        "data/origin/conll03/eng.testa",
        "data/origin/conll03/eng.testb"
    ]
    with jsonlines.open("data/origin/conll03/conll03.jsonl", "w") as writer:
        for sents, sents_labels in tqdm(ner_corpus_generator(filedirs, 50, 8, 0, 3, conll03_tag_to_spans)):
            trees = parser.predict([{"tokens": sent, "postags": ["PAD_TAG"]*len(sent)} for sent in sents])
            trees = [tree["pred_linearized_tree"] for tree in trees]

            for i in range(len(sents)):
                writer.write({
                    "tokens": sents[i],
                    "task": "named_entity_recognition",
                    "domain": "conll03",
                    "language": "en",
                    "linearized_tree": trees[i],
                    "ner_label": sents_labels[i],
                })


def ner_preprocess(model_dir: str, cuda_device: int):
    archive = load_archive(model_dir, cuda_device=cuda_device)
    parser = BaseConstituencyParserPredictor(archive.model, archive.validation_dataset_reader)
    conll03_preprocess(parser)
    mit_restaurant_preprocess(parser)


def build_joint_lm_cp_ner_dataset1():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner/train1.jsonl", "w") as writer:
        writer.write_all(cp_data)

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)


def build_joint_lm_cp_ner_dataset2():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner/train2.jsonl", "w") as writer:

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

            writer.write_all(cp_data)

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)
        writer.write_all(cp_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)
        writer.write_all(cp_data)


def build_joint_lm_cp_ner_dataset3():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner/train3.jsonl", "w") as writer:

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

            writer.write_all(cp_data)

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)
        writer.write_all(cp_data)
        for line in conll_data:
            line["task"] = "language_model"
            line.pop("ner_label")
            writer.write(line)
        writer.write_all(cp_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)
        writer.write_all(cp_data)
        for line in mitrest_data:
            line["task"] = "language_model"
            line.pop("ner_label")
            writer.write(line)
        writer.write_all(cp_data)


def ccg_corpus_generator(files: List[str]):
    snts, snts_labels = [], []
    tokens, labels = [], []

    for filedir in files:
        with open(filedir, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if line == "":
                    if 6 <= len(tokens) <= 200:
                        snts.append(tokens)
                        snts_labels.append(labels)
                    tokens, labels = [], []

                    if len(snts) >= 50:
                        yield snts, snts_labels
                        snts, snts_labels = [], []
                else:
                    line = line.split("|||")
                    if line[0] == "(":
                        line[0] = "-LRB-"
                    if line[0] == ")":
                        line[0] = "-RRB-"
                    if "(" in line[0]:
                        print(line[0])
                        line[0] = line[0].replace("(", "")
                    if ")" in line[0]:
                        print(line[0])
                        line[0] = line[0].replace(")", "")
                    tokens.append(line[0])
                    labels.append(line[1].replace(" ", ""))

    if len(snts) > 0:
        yield snts, snts_labels


def ccg_preprocess(model_dir: str, cuda_device: int):
    archive = load_archive(model_dir, cuda_device=cuda_device)
    parser = BaseConstituencyParserPredictor(archive.model, archive.validation_dataset_reader)

    filedirs: str = [
        "data/origin/ccg/ccgbank/ccg.train",
        "data/origin/ccg/ccgbank/ccg.dev",
        "data/origin/ccg/ccgbank/ccg.test"
    ]
    with jsonlines.open("data/origin/ccg/ccgbank/ccg.jsonl", "w") as writer:
        for sents, sents_labels in tqdm(ccg_corpus_generator(filedirs)):
            trees = parser.predict([{"tokens": sent, "postags": ["PAD_TAG"]*len(sent)} for sent in sents])
            trees = [tree["pred_linearized_tree"] for tree in trees]

            for i in range(len(sents)):
                writer.write({
                    "tokens": sents[i],
                    "task": "ccg_parsing",
                    "domain": "ccg_domain",
                    "language": "en",
                    "linearized_tree": trees[i],
                    "ccg_label": sents_labels[i],
                })


def build_joint_lm_cp_ner_ccg_dataset1():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner_ccg/train1.jsonl", "w") as writer:
        writer.write_all(cp_data)

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)

        ccg_data = []
        with jsonlines.open("data/origin/ccg/ccgbank/ccg.jsonl", "r") as reader:
            for line in reader:
                ccg_data.append(line)
        shuffle(ccg_data)
        ccg_data = ccg_data[:lm_data_num]
        writer.write_all(ccg_data)


def build_joint_lm_cp_ner_ccg_dataset2():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner_ccg/train2.jsonl", "w") as writer:

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

            writer.write_all(cp_data)

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)
        writer.write_all(cp_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)
        writer.write_all(cp_data)

        ccg_data = []
        with jsonlines.open("data/origin/ccg/ccgbank/ccg.jsonl", "r") as reader:
            for line in reader:
                ccg_data.append(line)
        shuffle(ccg_data)
        ccg_data = ccg_data[:lm_data_num]
        writer.write_all(ccg_data)
        writer.write_all(cp_data)


def build_joint_lm_cp_ner_ccg_dataset4():
    lm_data_num = 15000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner_ccg/train5.jsonl", "w") as writer:

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

            writer.write_all(cp_data)

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)
        writer.write_all(cp_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)
        writer.write_all(cp_data)

        ccg_data = []
        with jsonlines.open("data/origin/ccg/ccgbank/ccg.jsonl", "r") as reader:
            for line in reader:
                ccg_data.append(line)
        shuffle(ccg_data)
        ccg_data = ccg_data[:lm_data_num]
        writer.write_all(ccg_data)
        writer.write_all(cp_data)


def build_joint_lm_cp_ner_ccg_dataset3():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ner_ccg/train3.jsonl", "w") as writer:

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

            writer.write_all(cp_data)

        conll_data = []
        with jsonlines.open("data/origin/conll03/conll03.jsonl", "r") as reader:
            for line in reader:
                conll_data.append(line)
        shuffle(conll_data)
        conll_data = conll_data[:lm_data_num]
        writer.write_all(conll_data)
        writer.write_all(cp_data)
        for line in conll_data:
            line["task"] = "language_model"
            line.pop("ner_label")
            writer.write(line)
        writer.write_all(cp_data)

        mitrest_data = []
        with jsonlines.open("data/origin/mit_restaurant/mitrest.jsonl", "r") as reader:
            for line in reader:
                mitrest_data.append(line)
        shuffle(mitrest_data)
        mitrest_data = mitrest_data[:lm_data_num]
        writer.write_all(mitrest_data)
        writer.write_all(cp_data)
        for line in mitrest_data:
            line["task"] = "language_model"
            line.pop("ner_label")
            writer.write(line)
        writer.write_all(cp_data)

        ccg_data = []
        with jsonlines.open("data/origin/ccg/ccgbank/ccg.jsonl", "r") as reader:
            for line in reader:
                ccg_data.append(line)
        shuffle(ccg_data)
        ccg_data = ccg_data[:lm_data_num]
        writer.write_all(ccg_data)
        writer.write_all(cp_data)
        for line in ccg_data:
            line["task"] = "language_model"
            line.pop("ccg_label")
            writer.write(line)
        writer.write_all(cp_data)

        cp_lm_data = [line for line in cp_data if len(line["tokens"]) >= 6]
        shuffle(cp_lm_data)
        cp_lm_data = cp_lm_data[:lm_data_num]
        for line in cp_lm_data:
            line["task"] = "language_model"
            writer.write(line)


def build_joint_lm_cp_ner_ccg_domain_dataset():
    for i in range(5, 6):
        with jsonlines.open(f"data/joint_lm_cp_ner_ccg/train{i}.jsonl", "r") as reader, jsonlines.open(f"data/joint_lm_cp_ner_ccg/train{i}_domain.jsonl", "w") as writer:
            for line in reader:
                if line["domain"] == "ccg_domain":
                    line["domain"] = "news"
                writer.write(line)


def build_joint_lm_cp_ccg_dataset2():
    lm_data_num = 10000

    cp_data = []
    with jsonlines.open("data/base/consti_parsing/train.jsonl", "r") as reader:
        for line in reader:
            cp_data.append(line)

    with jsonlines.open("data/joint_lm_cp_ccg/train2.jsonl", "w") as writer:

        domains = ["dialogue", "forum", "law", "literature", "review"]
        for domain in domains:
            lm_domain_data = []
            with open(f"data/origin/raw_corpus/{domain}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    lm_domain_data.append(line)

            shuffle(lm_domain_data)
            lm_domain_data = lm_domain_data[:lm_data_num]
            for sent in lm_domain_data:
                tree = parse_bracketed_parse_tree(sent)
                tokens = list(tree.leaves())
                writer.write({
                    "tokens": tokens,
                    "task": "language_model",
                    "domain": domain,
                    "language": "en",
                    "linearized_tree": sent
                })

            writer.write_all(cp_data)

        ccg_data = []
        with jsonlines.open("data/origin/ccg/ccgbank/ccg.jsonl", "r") as reader:
            for line in reader:
                ccg_data.append(line)
        shuffle(ccg_data)
        ccg_data = ccg_data[:lm_data_num]
        for inst in ccg_data:
            inst["domain"] = "news"
        writer.write_all(ccg_data)
        writer.write_all(cp_data)


if __name__ == "__main__":
    # build_base_cp_dataset()

    # domains = ["dialogue", "forum", "law", "literature", "review"]
    # for domain in domains:
    #     clean_and_parse_raw_corpus(
    #         input_file=f"./data/origin/raw_corpus/{domain}_10w.txt",
    #         output_file=f"./data/origin/raw_corpus/{domain}.tag.txt",
    #         model_dir="./results/base/v2_3e_sub/"
    #     )

    # domains = ["dialogue", "forum", "law", "literature", "review"]
    # for domain in domains:
    #     for num in [10000]:
    #         build_joint_lm_cp_dataset(
    #             cp_path="./data/base/consti_parsing/train.jsonl",
    #             lm_path=f"./data/origin/raw_corpus/{domain}.tag.txt",
    #             output_path=f"./data/joint_lm_cp/train_{domain}_{num}.jsonl",
    #             num_lm=num,
    #             domain=domain
    #         )

    # ner_preprocess(
    #     model_dir="./results/base/v2_3e_sub/", cuda_device=0
    # )

    # build_joint_lm_cp_ner_dataset()

    # add_ptb_language_model_dataset()

    # add_ner_language_model_dataset()

    # ccg_preprocess(model_dir="./results/base/v2_3e_sub/", cuda_device=0)

    # build_joint_lm_cp_ner_dataset1()
    # build_joint_lm_cp_ner_dataset2()
    # build_joint_lm_cp_ner_dataset3()
    # build_joint_lm_cp_ner_ccg_dataset1()
    # build_joint_lm_cp_ner_ccg_dataset2()
    # build_joint_lm_cp_ner_ccg_dataset3()
    build_joint_lm_cp_ner_ccg_dataset4()
    build_joint_lm_cp_ner_ccg_domain_dataset()
    # build_joint_lm_cp_ccg_dataset2()
