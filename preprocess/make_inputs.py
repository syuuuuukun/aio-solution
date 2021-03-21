import json
import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool

from typing import List
from io import open
import gzip

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForMultipleChoice,
    BertJapaneseTokenizer,
    PreTrainedTokenizer,
)

import math
import MeCab
from collections import Counter, defaultdict
import pickle
import time
import unidic

import logging
logger = logging.getLogger(__name__)

tagger = MeCab.Tagger('-d "{}"'.format(unidic.DICDIR))
STOP_POSTAGS = ('BOS/EOS',"代名詞","接続詞","感動詞","動詞,非自立可能","助動詞",'助詞',"接頭辞","記号,一般","補助記号","空白")
SEPARATE_TOKEN = '。'


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings,ctx1,ctx2,ctx3,label=None):
        """Constructs a InputExample.
        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence
                      (context of corresponding question).
            question: string. The untokenized text of the second sequence
                      (question).
            endings: list of str. multiple choice's options.
                     Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.ctx1 = ctx1
        self.ctx2 = ctx2
        self.ctx3 = ctx3


class InputFeatures(object):
    def __init__(self, example_id, choices_features1,choices_features2,choices_features3,choices_features4, label):
        self.example_id = example_id
        self.choices_features1 = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features1
        ]
        self.choices_features2 = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features2
        ]
        self.choices_features3 = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features3
        ]
        self.choices_features4 = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features4
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_examples(self, mode, data_dir, fname, entities_fname):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
        
class JaqketProcessor(DataProcessor):

    def _get_entities(self, data_dir, entities_fname):
        logger.info("LOOKING AT {} entities".format(data_dir))
        entities = dict()
        for line in self._read_json_gzip(os.path.join(data_dir, entities_fname)):
            entity = json.loads(line.strip())
            entities[entity["title"]] = entity["text"]

        return entities

    def get_examples(self, mode, data_dir, json_data, entities, num_options=20):
        """See base class."""
        logger.info("LOOKING AT {} [{}]".format(data_dir, mode))
        entities = entities
        return self._create_examples(
            json_data,
            mode,
            entities,
            num_options,
        )

    def get_labels(self):
        """See base class."""
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
        ]

    def _read_json(self, input_file):
        return input_file
#         with open(input_file, "r", encoding="utf-8") as fin:
#             lines = fin.readlines()
#             return lines

    def _read_json_gzip(self, input_file):
        with gzip.open(input_file, "rt", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, t_type, entities, num_options):
        """Creates examples for the training and dev sets."""

        examples = []
        skip_examples = 0

        # for line in tqdm.tqdm(
        #    lines, desc="read jaqket data", ascii=True, ncols=80
        # ):
        logger.info("read jaqket data: {}".format(len(lines)))
        for line in lines:
            data_raw = line

            id = data_raw["qid"]
            question = data_raw["question"].replace("_", "")  # "_" は cloze question
            options = data_raw["answer_candidates"][:num_options]  # TODO
            answer = data_raw["answer_entity"]
            ctx1 = data_raw["ctx1"]
            ctx2 = data_raw["ctx2"]
            ctx3 = data_raw["ctx3"]

            if answer not in options:
                continue

            if len(options) != num_options:
                skip_examples += 1
                continue

            contexts = [entities[options[i]] for i in range(num_options)]
            truth = str(options.index(answer))

            if len(options) == num_options:  # TODO
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=contexts,
                        endings=options,
                        ctx1=ctx1,
                        ctx2=ctx2,
                        ctx3=ctx3,
                        label=truth,
                    )
                )

        if t_type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None

        logger.info("len examples: {}".format(len(examples)))
        logger.info("skip examples: {}".format(skip_examples))

        return examples
    
def convert_examples_to_features(example):
#     tokenizer: PreTrainedTokenizer,)
    
    
    label_list = [f"{i}" for i in range(20)]
    label_map = {label: i for i, label in enumerate(label_list)}
    pad_token_segment_id=0
    pad_on_left=False
    pad_token=0
    mask_padding_with_zero=True
    max_length = 768
    
    contexts,endings,question,label,example_id,ctx_add1,ctx_add2,ctx_add3 = example
    
    ##top1_ignore-answer
    entity_text1 = "。".join([entities[doc_id2title[s[0]]] for s in ctx_add1[:1]])
    ##top5_in-answer
    entity_text2 = "。".join([entities[doc_id2title[s[0]]] for s in ctx_add2[:5]])
    
    features = []
    context2_1 = get_contexts_bm25(entity_text1,question)
    context2_3 = get_contexts_bm25(entity_text2,question)
    ##正解エンティティの本文 + 正解候補のタイトルを除外したBM25で引っ張ってきた文章(top1)
    choices_features1 = []
    ##選択肢本文のみ
    choices_features2 = []
    ##BM25で引っ張ってきた文章のみ(top5)
    choices_features3 = []
    ##BM25で引っ張ってきた文章のみ(top5)(wikiを検索するときも並び替えの時もqueryに選択肢を追加)
    choices_features4 = []
    for ending_idx, (context, ending) in enumerate(
        zip(contexts,endings)
    ):
        input_ids, attention_mask, token_type_ids = make_bert_input1(ending,question,context2_1,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id)
        choices_features1.append((input_ids, attention_mask, token_type_ids))
        input_ids, attention_mask, token_type_ids = make_bert_input2(ending,question,context2_1,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id)
        choices_features2.append((input_ids, attention_mask, token_type_ids))
        input_ids, attention_mask, token_type_ids = make_bert_input3(ending,question,context2_3,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id)
        choices_features3.append((input_ids, attention_mask, token_type_ids))
        
        
        entity_text = "。".join([entities[doc_id2title[s[0]]] for s in ctx_add3[ending_idx][:5]])
        context2_4 = get_contexts_bm25_add_answer(entity_text,question,ending)
        input_ids, attention_mask, token_type_ids = make_bert_input3(ending,question,context2_4,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id)
        choices_features4.append((input_ids, attention_mask, token_type_ids))


    label = label_map[label]

    features.append(
        InputFeatures(
            example_id=example_id,
            choices_features1=choices_features1,
            choices_features2=choices_features2,
            choices_features3=choices_features3,
            choices_features4=choices_features4,
            label=label,
        )
    )

    return features

def make_bert_input1(ending,question,context2,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id):
    context1 = get_contexts_bm25(entities[ending],question)
    text_a = context1[:768]+ tokenizer.sep_token + context2
    text_b = question + tokenizer.sep_token + ending

    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation="only_first",  # 常にcontextをtruncate
    )

    input_ids, token_type_ids = (
        inputs["input_ids"],
        inputs["token_type_ids"],
    )

    # The mask has 1 for real tokens and 0 for padding tokens. Only
    # real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + attention_mask
        token_type_ids = (
            [pad_token_segment_id] * padding_length
        ) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + (
            [pad_token_segment_id] * padding_length
        )
    return input_ids, attention_mask, token_type_ids

def make_bert_input2(ending,question,context2,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id):
    context1 = get_contexts_bm25(entities[ending],question)
    text_a = context1
    text_b = question + tokenizer.sep_token + ending

    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation="only_first",  # 常にcontextをtruncate
    )

    input_ids, token_type_ids = (
        inputs["input_ids"],
        inputs["token_type_ids"],
    )

    # The mask has 1 for real tokens and 0 for padding tokens. Only
    # real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + attention_mask
        token_type_ids = (
            [pad_token_segment_id] * padding_length
        ) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + (
            [pad_token_segment_id] * padding_length
        )
    return input_ids, attention_mask, token_type_ids

def make_bert_input3(ending,question,context2,mask_padding_with_zero,max_length,pad_on_left,pad_token,pad_token_segment_id):
    context1 = context2
    text_a = context1
    text_b = question + tokenizer.sep_token + ending

    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation="only_first",  # 常にcontextをtruncate
    )

    input_ids, token_type_ids = (
        inputs["input_ids"],
        inputs["token_type_ids"],
    )

    # The mask has 1 for real tokens and 0 for padding tokens. Only
    # real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + attention_mask
        token_type_ids = (
            [pad_token_segment_id] * padding_length
        ) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + (
            [pad_token_segment_id] * padding_length
        )
    return input_ids, attention_mask, token_type_ids


def get_qus_answers(input_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()    
    queries = []
    answers = []
    for line in tqdm(lines):
        data_raw = json.loads(line.strip("\n"))
        question = data_raw["question"].replace("_", "")  # "_" は cloze question
        answer = data_raw['answer_candidates']
        queries += [(question,answer)]
#         answers += [answer]
    return queries

def read_json(x):
    with open(x, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        lines = [eval(line) for line in lines]    
    return lines

def get_contexts_bm25(sentence_list,query,topk=1000):
    sentence_list = sentence_list.split("。")
    inverted_index = defaultdict(list)
    sentence_id2sentence = [sentence for sentence in sentence_list]
    sentence_id2token_count = []
    for sentence_id, sentence in enumerate(sentence_list):
        tokens = parse_text(sentence)
    
        sentence_id2token_count += [len(tokens)]

        count_tokens = Counter(tokens)
        for token, count in count_tokens.items():
            inverted_index[token] += [(sentence_id, count)]

    avgdl = sum(sentence_id2token_count) / len(sentence_id2token_count)
    parsed_query = parse_text(query)
    target_posting = {}
    for token in parsed_query:
        if token in inverted_index:
            postings_list = inverted_index[token]
            target_posting[token] = postings_list

    # bm25スコアでor検索
    k1 = 2.0
    b = 0.75
    all_docs = len(sentence_list)
    sentence_id2tfidf = [0 for i in range(all_docs)]
    for token, postings_list in target_posting.items():
        idf = math.log2((all_docs-len(postings_list)+0.5) / (len(postings_list) + 0.5))
        # idfが負になる単語は一般的すぎるので無視
        idf = max(idf, 0)
        if idf == 0:
            continue
        for sentence_id, tf in postings_list:
            dl = sentence_id2token_count[sentence_id]
            token_tfidf = idf * ((tf * (k1 + 1))/(tf + k1 * (1-b+b*(dl/avgdl))))
            sentence_id2tfidf[sentence_id] += token_tfidf

    sentences = [(sentence_id, tfidf) for sentence_id, tfidf in enumerate(sentence_id2tfidf) if tfidf != 0]
    sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
    return "。".join(list(map(lambda x: sentence_id2sentence[x[0]], sentences[:topk])))

def get_contexts_bm25_add_answer(sentence_list,query,answer,topk=1000):
    sentence_list = sentence_list.split("。")
    inverted_index = defaultdict(list)
    sentence_id2sentence = [sentence for sentence in sentence_list]
    sentence_id2token_count = []
    for sentence_id, sentence in enumerate(sentence_list):
        tokens = parse_text(sentence)
    
        sentence_id2token_count += [len(tokens)]

        count_tokens = Counter(tokens)
        for token, count in count_tokens.items():
            inverted_index[token] += [(sentence_id, count)]

    avgdl = sum(sentence_id2token_count) / len(sentence_id2token_count)
    parsed_query = parse_text(query)
    parsed_query += parse_text(answer)
    target_posting = {}
    for token in parsed_query:
        if token in inverted_index:
            postings_list = inverted_index[token]
            target_posting[token] = postings_list

    # bm25スコアでor検索
    k1 = 2.0
    b = 0.75
    all_docs = len(sentence_list)
    sentence_id2tfidf = [0 for i in range(all_docs)]
    for token, postings_list in target_posting.items():
        idf = math.log2((all_docs-len(postings_list)+0.5) / (len(postings_list) + 0.5))
        # idfが負になる単語は一般的すぎるので無視
        idf = max(idf, 0)
        if idf == 0:
            continue
        for sentence_id, tf in postings_list:
            dl = sentence_id2token_count[sentence_id]
            token_tfidf = idf * ((tf * (k1 + 1))/(tf + k1 * (1-b+b*(dl/avgdl))))
            sentence_id2tfidf[sentence_id] += token_tfidf

    sentences = [(sentence_id, tfidf) for sentence_id, tfidf in enumerate(sentence_id2tfidf) if tfidf != 0]
    sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
    return "。".join(list(map(lambda x: sentence_id2sentence[x[0]], sentences[:topk])))

def parse_text(text):
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.feature.startswith(STOP_POSTAGS):
            pass
        else:
            feature = node.feature.split(",")
            if len(feature) >7:
                tokens += [feature[7].lower()]
            else:
                tokens += [node.surface.lower()]
        node = node.next
    return tokens


def select_field1(features, field):
    return [
        [choice[field] for choice in feature.choices_features1] for feature in features
    ]

def select_field2(features, field):
    return [
        [choice[field] for choice in feature.choices_features2] for feature in features
    ]

def select_field3(features, field):
    return [
        [choice[field] for choice in feature.choices_features3] for feature in features
    ]

def select_field4(features, field):
    return [
        [choice[field] for choice in feature.choices_features4] for feature in features
    ]

def get_batch(features):
    all_input_ids1 = torch.tensor(select_field1(features, "input_ids"), dtype=torch.long)
    all_input_mask1 = torch.tensor(select_field1(features, "input_mask"), dtype=torch.long)
    all_segment_ids1 = torch.tensor(select_field1(features, "segment_ids"), dtype=torch.long)
    all_label_ids1 = torch.tensor([f.label for f in features], dtype=torch.long)  
    
    all_input_ids2 = torch.tensor(select_field2(features, "input_ids"), dtype=torch.long)
    all_input_mask2 = torch.tensor(select_field2(features, "input_mask"), dtype=torch.long)
    all_segment_ids2 = torch.tensor(select_field2(features, "segment_ids"), dtype=torch.long)
    all_label_ids2 = torch.tensor([f.label for f in features], dtype=torch.long) 
    
    all_input_ids3 = torch.tensor(select_field3(features, "input_ids"), dtype=torch.long)
    all_input_mask3 = torch.tensor(select_field3(features, "input_mask"), dtype=torch.long)
    all_segment_ids3 = torch.tensor(select_field3(features, "segment_ids"), dtype=torch.long)
    all_label_ids3 = torch.tensor([f.label for f in features], dtype=torch.long) 
    
    all_input_ids4 = torch.tensor(select_field4(features, "input_ids"), dtype=torch.long)
    all_input_mask4 = torch.tensor(select_field4(features, "input_mask"), dtype=torch.long)
    all_segment_ids4 = torch.tensor(select_field4(features, "segment_ids"), dtype=torch.long)
    all_label_ids4 = torch.tensor([f.label for f in features], dtype=torch.long) 
    
    inputs1 = (all_input_ids1,all_input_mask1,all_segment_ids1,all_label_ids1)
    inputs2 = (all_input_ids2,all_input_mask2,all_segment_ids2,all_label_ids2)
    inputs3 = (all_input_ids3,all_input_mask3,all_segment_ids3,all_label_ids3)
    inputs4 = (all_input_ids4,all_input_mask4,all_segment_ids4,all_label_ids4)
    
    return inputs1,inputs2,inputs3,inputs4

def get_inputs(mode="train"):
    root_path = "../data/"
    json_file = f"{mode}_questions.json"
    
    json_data = read_json(root_path+json_file)
    ctx1 = pickle.load(open(root_path+f"{mode}_ctx_ids-top10.pkl","rb"))
    ctx2 = pickle.load(open(root_path+f"{mode}_ctx_ids-top10_ignore-answers.pkl","rb"))
    ctx3 = pickle.load(open(root_path+f"{mode}_ctx_ids-top10_query-add-answers.pkl","rb"))
    
    
    for data,c1,c2,c3 in zip(json_data,ctx1,ctx2,ctx3):
        data["ctx1"] = c2
        data["ctx2"] = c1
        data["ctx3"] = c3
        
    processor = JaqketProcessor()
    examples  = processor.get_examples("dev",root_path,json_data,entities)
    values = [(ex.contexts,ex.endings,ex.question,ex.label,ex.example_id,ex.ctx1,ex.ctx2,ex.ctx3) for ex in examples]
    with Pool(multiprocessing.cpu_count()) as p:
        features = list(tqdm(p.imap(convert_examples_to_features,values), total=len(values)))
        features = [f[0] for f in features]
    
    
    batch1,batch2,batch3,batch4 = get_batch(features)
    
    torch.save({f"{mode}_input_ids":batch1[0],
                f"{mode}_input_mask":batch1[1],
                f"{mode}_segment_ids":batch1[2],
                f"{mode}_label_ids":batch1[3]},root_path+f"basev2-{mode}_features-seq768-sorted_title-bm25_search-search_ver3.pt")
    
    torch.save({f"{mode}_input_ids":batch2[0],
                f"{mode}_input_mask":batch2[1],
                f"{mode}_segment_ids":batch2[2],
                f"{mode}_label_ids":batch2[3]},root_path+f"basev2-{mode}_features-seq768-title_only-search_ver3.pt")
    
    torch.save({f"{mode}_input_ids":batch3[0],
                f"{mode}_input_mask":batch3[1],
                f"{mode}_segment_ids":batch3[2],
                f"{mode}_label_ids":batch3[3]},root_path+f"basev2-{mode}_features-seq768-question_only-search_ver3.pt")
    
    torch.save({f"{mode}_input_ids":batch4[0],
                f"{mode}_input_mask":batch4[1],
                f"{mode}_segment_ids":batch4[2],
                f"{mode}_label_ids":batch4[3]},root_path+f"basev2-{mode}_features-seq768-question_only-add_answer-search_ver3.pt")
    
    
    return features


if __name__ == "__main__":
    
    with open('../ir_dump/doc_id2title.pickle', 'rb') as f:
        doc_id2title = pickle.load(f)

    input_file = '../data/all_entities.json.gz'
    entitie2id = {k:v for v,k in enumerate(doc_id2title)}
    with gzip.open(input_file, "rt", encoding="utf-8") as fin:
        lines = fin.readlines()

    entities = dict()
    for line in lines:
        entity = json.loads(line.strip())
        entities[entity["title"]] = entity["text"]
    del lines    
    
    path_name = "cl-tohoku/bert-base-japanese-v2"
    tokenizer = BertJapaneseTokenizer.from_pretrained(path_name)

    features = get_inputs(mode="train")
    features = get_inputs(mode="dev1")
    features = get_inputs(mode="dev2")