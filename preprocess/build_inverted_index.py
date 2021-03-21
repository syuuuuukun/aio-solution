import gzip
import json

import pickle
import time
import gc
import os
from multiprocessing import Pool,cpu_count

import MeCab
import unidic

from contextlib import ExitStack
from collections import defaultdict, Counter
from tqdm import tqdm


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

def get_inverted_index(entity):
    title, text = entity
    tokens = parse_text(text)
    token_len = len(tokens)
    count_tokens = Counter(tokens)
    return title,token_len,count_tokens

def load_index_line(index_line):
    if index_line == '':
        return []
    return list(map(lambda x: tuple(map(int, x.split(':'))), index_line.split(' ')))

if __name__ == "__main__":
    input_file = '../data/all_entities.json.gz'
    tagger = MeCab.Tagger('-d "{}"'.format(unidic.DICDIR))
    STOP_POSTAGS = ('BOS/EOS',"代名詞","接続詞","感動詞","動詞,非自立可能","助動詞",'助詞',"接頭辞","記号,一般","補助記号","空白")
    
    path = f"../ir_dump/"
    cpu_num = cpu_count()
#     cpu_num = 36
    os.makedirs(path,exist_ok=True)
    
    with gzip.open(input_file, "rt", encoding="utf-8") as fin:
        lines = fin.readlines()
        
    entities = dict()
    for line in lines:
        entity = json.loads(line.strip())
        entities[entity["title"]] = entity["text"]
    del lines
    gc.collect()
    
    # 文書ごとのtokenをマルチプロセスで計算する
    entity_len = len(entities.items())
    with Pool(cpu_num) as p:
        imap = p.imap(get_inverted_index,entities.items())
        result = list(tqdm(imap, total=entity_len))
        
        
    # 転置インデックス分割作成
    partial_size = 10**5
    inverted_index = defaultdict(list)
    doc_id2title = []
    doc_id2token_count = []
    for doc_id, (title,token_len,count_tokens) in tqdm(enumerate(result), total=len(result)):
        doc_id2title += [title]
        doc_id2token_count += [token_len]
        for token, count in count_tokens.items():
            inverted_index[token] += [(doc_id, count)]

        if (doc_id + 1) % partial_size == 0:
            sorted_vocab = sorted(inverted_index.keys())
            partial_id = doc_id // partial_size

            with open(path+'partial_dict_{}'.format(partial_id), 'w', encoding='utf-8') as fout:
                for token in sorted_vocab:
                    fout.write(token + '\n')

            with open(path+'partial_inverted_index_{}'.format(partial_id), 'w', encoding='utf-8') as fout:
                for token in sorted_vocab:
                    posting_list = ' '.join([str(doc_id)+':'+str(tf)for doc_id, tf in inverted_index[token]])
                    fout.write(posting_list + '\n')
            inverted_index = defaultdict(list)
    sorted_vocab = sorted(inverted_index.keys())
    partial_id = (len(entities)-1) // partial_size

    with open(path+'partial_dict_{}'.format(partial_id), 'w', encoding='utf-8') as fout:
        for token in sorted_vocab:
            fout.write(token + '\n')

    with open(path+'partial_inverted_index_{}'.format(partial_id), 'w', encoding='utf-8') as fout:
        for token in sorted_vocab:
            posting_list = ' '.join([str(doc_id)+':'+str(tf)for doc_id, tf in inverted_index[token]])
            fout.write(posting_list + '\n')

    # docment_idをタイトルやトークン数に変換するlistを保存
    with open(path+'doc_id2title.pickle', 'wb') as f:
        pickle.dump(doc_id2title, f)

    with open(path+'doc_id2token_count.pickle', 'wb') as f:
        pickle.dump(doc_id2token_count, f)
    
    
    # 分割転置インデックスのマージ
    start_time = time.time()
    dict_filenames = [path+'partial_dict_{}'.format(partial_id) for partial_id in range(10)]
    index_filenames = [path+'partial_inverted_index_{}'.format(partial_id) for partial_id in range(10)]
    line2token = []

    with ExitStack() as stack, open(path+'inverted_index', 'w', encoding='utf-8') as fout:
        dict_files = [stack.enter_context(open(fname, 'r', encoding='utf-8')) for fname in dict_filenames]
        index_files = [stack.enter_context(open(fname, 'r', encoding='utf-8')) for fname in index_filenames]
        tokens = []
        postings = []
        for dict_file, index_file in zip(dict_files, index_files):
            token = dict_file.readline().rstrip()
            index_line = index_file.readline().rstrip()
            partial_posting_list = load_index_line(index_line)
            tokens += [token]
            postings += [partial_posting_list]

        while sorted_token := sorted(filter(lambda x: x != '', tokens)):
            top_token = sorted_token[0]
            posting_list = []
            for partial_id, (dict_file, index_file) in enumerate(zip(dict_files, index_files)):
                token = tokens[partial_id]
                if token == top_token:
                    posting_list += postings[partial_id]
                    token = dict_file.readline().rstrip()
                    index_line = index_file.readline().rstrip()
                    partial_posting_list = load_index_line(index_line)
                    tokens[partial_id] = token
                    postings[partial_id] = partial_posting_list

            posting_list = ' '.join([str(doc_id)+':'+str(tf) for doc_id, tf in posting_list])
            line2token += [top_token]
            fout.write(posting_list + '\n')

    end_time = time.time() - start_time
    print(end_time)

    # トークンと転置インデックスファイルのポインタ対応づけ
    token2pointer = {}
    with open(path+'inverted_index', 'r', encoding='utf-8') as fin:
        for token in tqdm(line2token):
            start = fin.tell()
            line = fin.readline()
            end = fin.tell()
            token2pointer[token] = (start, end)

    with open(path+'token2pointer.pickle', 'wb') as f:
        pickle.dump(token2pointer, f)