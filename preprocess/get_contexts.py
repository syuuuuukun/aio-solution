import gzip
import json
import pickle
import time

import MeCab
import unidic

import math
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

tagger = MeCab.Tagger('-d "{}"'.format(unidic.DICDIR))
STOP_POSTAGS = ('BOS/EOS',"代名詞","接続詞","感動詞","動詞,非自立可能","助動詞",'助詞',"接頭辞","記号,一般","補助記号","空白")

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

def load_index_line(index_line):
    return list(map(lambda x: tuple(map(int, x.split(':'))), index_line.split(' ')))

def search_entity_ignore_answer(querys, topk=10):
    query,answer_candidates = querys
    ignore_docid = [entitie2id[answer] for answer in answer_candidates]
    
    avgdl = sum(doc_id2token_count) / len(doc_id2token_count)
    parsed_query = parse_text(query)
    target_posting = {}
    with open('../ir_dump/inverted_index', 'r', encoding='utf-8') as index_file:
        for token in parsed_query:
            if token in token2pointer:
                pointer, offset = token2pointer[token]
                index_file.seek(pointer)
                index_line = index_file.read(offset-pointer).rstrip()
                postings_list = load_index_line(index_line)
                target_posting[token] = postings_list

    # bm25スコアでor検索
    k1 = 2.0
    b = 0.75
    all_docs = len(entities)
    doc_id2tfidf = [0 for i in range(all_docs)]
    for token, postings_list in target_posting.items():
        idf = math.log2((all_docs-len(postings_list)+0.5) / (len(postings_list) + 0.5))
        # idfが負になる単語は一般的すぎるので無視
        idf = max(idf, 0)
        if idf == 0:
            continue
        for doc_id, tf in postings_list:
            dl = doc_id2token_count[doc_id]
            token_tfidf = idf * ((tf * (k1 + 1))/(tf + k1 * (1-b+b*(dl/avgdl))))
            doc_id2tfidf[doc_id] += token_tfidf
            
    for ignore_id in ignore_docid:
        doc_id2tfidf[ignore_id] = 0

    docs = [(doc_id, tfidf) for doc_id, tfidf in enumerate(doc_id2tfidf) if tfidf != 0]
    docs = sorted(docs, key=lambda x: x[1], reverse=True)
    return docs[:topk]

def search_entity(querys, topk=10):
    query,answer_candidates = querys
#     ignore_docid = [entitie2id[answer] for answer in answer_candidates]
    
    avgdl = sum(doc_id2token_count) / len(doc_id2token_count)
    parsed_query = parse_text(query)
    target_posting = {}
    with open('../ir_dump/inverted_index', 'r', encoding='utf-8') as index_file:
        for token in parsed_query:
            if token in token2pointer:
                pointer, offset = token2pointer[token]
                index_file.seek(pointer)
                index_line = index_file.read(offset-pointer).rstrip()
                postings_list = load_index_line(index_line)
                target_posting[token] = postings_list

    # bm25スコアでor検索
    k1 = 2.0
    b = 0.75
    all_docs = len(entities)
    doc_id2tfidf = [0 for i in range(all_docs)]
    for token, postings_list in target_posting.items():
        idf = math.log2((all_docs-len(postings_list)+0.5) / (len(postings_list) + 0.5))
        # idfが負になる単語は一般的すぎるので無視
        idf = max(idf, 0)
        if idf == 0:
            continue
        for doc_id, tf in postings_list:
            dl = doc_id2token_count[doc_id]
            token_tfidf = idf * ((tf * (k1 + 1))/(tf + k1 * (1-b+b*(dl/avgdl))))
            doc_id2tfidf[doc_id] += token_tfidf
            
#     for ignore_id in ignore_docid:
#         doc_id2tfidf[ignore_id] = 0

    docs = [(doc_id, tfidf) for doc_id, tfidf in enumerate(doc_id2tfidf) if tfidf != 0]
    docs = sorted(docs, key=lambda x: x[1], reverse=True)
    return docs[:topk]

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


def search_entity_candidates(querys, topk=10):
    query,candidates = querys
    
    avgdl = sum(doc_id2token_count) / len(doc_id2token_count)
    parsed_query = parse_text(query)
    target_posting = {}
    with open('../ir_dump/inverted_index', 'r', encoding='utf-8') as index_file:
        for token in parsed_query:
            if token in token2pointer:
                pointer, offset = token2pointer[token]
                index_file.seek(pointer)
                index_line = index_file.read(offset-pointer).rstrip()
                postings_list = load_index_line(index_line)
                target_posting[token] = postings_list

    # bm25スコアでor検索
    k1 = 2.0
    b = 0.75
    all_docs = len(entities)
    doc_id2tfidf = [0 for i in range(all_docs)]
    for token, postings_list in target_posting.items():
        idf = math.log2((all_docs-len(postings_list)+0.5) / (len(postings_list) + 0.5))
        # idfが負になる単語は一般的すぎるので無視
        idf = max(idf, 0)
        if idf == 0:
            continue
        for doc_id, tf in postings_list:
            dl = doc_id2token_count[doc_id]
            token_tfidf = idf * ((tf * (k1 + 1))/(tf + k1 * (1-b+b*(dl/avgdl))))
            doc_id2tfidf[doc_id] += token_tfidf
    
    # candidateごとの検索
    search_results = []
    with open('../ir_dump/inverted_index', 'r', encoding='utf-8') as index_file:
        for candidate in candidates:
            parsed_candidate = parse_text(candidate)
            
            candidate_target_posting = {}
            for token in parsed_candidate:
                if token in token2pointer:
                    pointer, offset = token2pointer[token]
                    index_file.seek(pointer)
                    index_line = index_file.read(offset-pointer).rstrip()
                    postings_list = load_index_line(index_line)
                    candidate_target_posting[token] = postings_list
                    
            candidate_tfidf = []
            # candidateとなる文字列が含まれるdoc_idの集合
            candidate_doc_ids = set()
            for token_position, (token, postings_list) in enumerate(candidate_target_posting.items()):
                idf = math.log2((all_docs-len(postings_list)+0.5) / (len(postings_list) + 0.5))
                # idfが負になる単語は一般的すぎるので無視
                idf = max(idf, 0)
                if idf == 0:
                    continue
                token_doc_ids = []
                for doc_id, tf in postings_list:
                    dl = doc_id2token_count[doc_id]
                    token_tfidf = idf * ((tf * (k1 + 1))/(tf + k1 * (1-b+b*(dl/avgdl))))
                    doc_id2tfidf[doc_id] += token_tfidf
                    candidate_tfidf += [(doc_id, token_tfidf)]
                    token_doc_ids += [doc_id]
                
                if token_position == 0:
                    candidate_doc_ids |= set(token_doc_ids)
                else:
                    candidate_doc_ids &= set(token_doc_ids)

            docs = [(doc_id, doc_id2tfidf[doc_id]) for doc_id in candidate_doc_ids]
            docs = sorted(docs, key=lambda x: x[1], reverse=True)
            search_results += [docs[:topk]]
            for doc_id, tfidf in candidate_tfidf:
                doc_id2tfidf[doc_id] -= tfidf
            
    return search_results


if __name__ == "__main__":
    input_file = '../data/all_entities.json.gz'
    with gzip.open(input_file, "rt", encoding="utf-8") as fin:
        lines = fin.readlines()

    entities = dict()
    for line in lines:
        entity = json.loads(line.strip())
        entities[entity["title"]] = entity["text"]
    del lines

    with open('../ir_dump/doc_id2title.pickle', 'rb') as f:
        doc_id2title = pickle.load(f)
    with open('../ir_dump/doc_id2token_count.pickle', 'rb') as f:
        doc_id2token_count = pickle.load(f)
    with open('../ir_dump/token2pointer.pickle', 'rb') as f:
        token2pointer = pickle.load(f)

    entitie2id = {k:v for v,k in enumerate(doc_id2title)}
    
    
    train_queries = get_qus_answers('../data/train_questions.json')
    dev1_queries = get_qus_answers('../data/dev1_questions.json')
    dev2_queries = get_qus_answers('../data/dev2_questions.json')
    
    with Pool(cpu_count()) as p:
        train_results1 = list(tqdm(p.imap(search_entity_ignore_answer, train_queries), total=len(train_queries)))
        dev1_results1 = list(tqdm(p.imap(search_entity_ignore_answer, dev1_queries), total=len(dev1_queries)))
        dev2_results1 = list(tqdm(p.imap(search_entity_ignore_answer, dev2_queries), total=len(dev2_queries)))
        
    with Pool(cpu_count()) as p:
        train_results2 = list(tqdm(p.imap(search_entity, train_queries), total=len(train_queries)))
        dev1_results2 = list(tqdm(p.imap(search_entity, dev1_queries), total=len(dev1_queries)))
        dev2_results2 = list(tqdm(p.imap(search_entity, dev2_queries), total=len(dev2_queries)))
        
    with Pool(cpu_count()) as p:
        train_results3 = list(tqdm(p.imap(search_entity_candidates, train_queries), total=len(train_queries)))
        dev1_results3  = list(tqdm(p.imap(search_entity_candidates, dev1_queries), total=len(dev1_queries)))
        dev2_results3  = list(tqdm(p.imap(search_entity_candidates, dev2_queries), total=len(dev2_queries)))

    with open('../data/train_ctx_ids-top10_ignore-answers.pkl', 'wb') as f:
        pickle.dump(train_results1, f)
    with open('../data/dev1_ctx_ids-top10_ignore-answers.pkl', 'wb') as f:
        pickle.dump(dev1_results1, f)
    with open('../data/dev2_ctx_ids-top10_ignore-answers.pkl', 'wb') as f:
        pickle.dump(dev2_results1, f)
        
    with open('../data/train_ctx_ids-top10.pkl', 'wb') as f:
        pickle.dump(train_results2, f)
    with open('../data/dev1_ctx_ids-top10.pkl', 'wb') as f:
        pickle.dump(dev1_results2, f)
    with open('../data/dev2_ctx_ids-top10.pkl', 'wb') as f:
        pickle.dump(dev2_results2, f)
        
    with open('../data/train_ctx_ids-top10_query-add-answers.pkl', 'wb') as f:
        pickle.dump(train_results3, f)
    with open('../data/dev1_ctx_ids-top10_query-add-answers.pkl', 'wb') as f:
        pickle.dump(dev1_results3, f)
    with open('../data/dev2_ctx_ids-top10_query-add-answers.pkl', 'wb') as f:
        pickle.dump(dev2_results3, f)