import argparse
import mxnet as mx
import numpy as np

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, Block
import gluonnlp as nlp
import time
import random
from gluonnlp.data import BERTTokenizer

parser = argparse.ArgumentParser(description='Bert')
parser.add_argument("--file", type=str, help="the file with data")
parser.add_argument('--id', type=int, help='')
parser.add_argument('--name', type=str, help='')
args = parser.parse_args()
max_embs = 5000000

random.seed(123)
np.random.seed(123)
mx.random.seed(123)

dropout_prob = 0.1
ctx = mx.gpu(args.id)
bert_model, bert_vocab = nlp.model.get_model(name='bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True,
                                             ctx=ctx,
                                             use_pooler=True,
                                             use_decoder=False,
                                             use_classifier=False,
                                             dropout=dropout_prob,
                                             embed_dropout=dropout_prob)
tokenizer = BERTTokenizer(bert_vocab, lower=True)

abstract_emb = np.zeros((max_embs, 768), dtype=np.float32)

paper_map = []
fp = open(args.file, 'r')
start = time.time()
for i, line in enumerate(fp):
    paper_id, abstract = line.split('\t')
    paper_id = int(paper_id)
    tokens = tokenizer(abstract)
    if len(tokens) > 512:
        print('paper {} has strings with {} tokens'.format(paper_id, len(tokens)))
        tokens = tokens[0:512]
    
    token_ids = mx.nd.expand_dims(mx.nd.array(bert_vocab[[bert_vocab.cls_token] + tokens + [bert_vocab.sep_token]],
                                              dtype=np.int32, ctx=ctx), axis=0)
    token_types = mx.nd.ones_like(token_ids, ctx=ctx)
    _, sent_embedding = bert_model(token_ids, token_types)
    emb = sent_embedding.transpose().squeeze()
    
    abstract_emb[i] = emb.asnumpy()
    paper_map.append(paper_id)
    assert len(paper_map) == i + 1

    if (i + 1) % 1000 == 0:
        print('creating {} embeddings takes {} seconds'.format(i + 1, time.time() - start))
        start = time.time()
fp.close()
print('get Bert embeddings for {} strings'.format(len(paper_map)))

paper_map = np.array(paper_map, dtype=np.int64)
np.save('{}_emb_{}.npy'.format(args.name, args.id), abstract_emb[:len(paper_map)])
np.save('{}_map_{}.npy'.format(args.name, args.id), paper_map)
