import os
import argparse
import torch
import pickle
import torch
import spacy
import numpy as np

from flask import Flask, request, jsonify
from model import SimpleRNN

app = Flask(__name__)

args = argparse.Namespace()

print("Init vocab")

VOCAB_FILE = "data/vocab/vocab_100d.p"

with open(VOCAB_FILE, 'rb') as f:
    vocab = pickle.load(f, encoding='latin1')

args.vocab_size=len(vocab.word_list)


args.hidden=200
args.embedding_dim = 100
args.position_size = 500
args.position_dim = 50
args.word_input_size = 100
args.sent_input_size = 2 * args.hidden
args.word_LSTM_hidden_units = args.hidden
args.sent_LSTM_hidden_units = args.hidden
args.pretrained_embedding = vocab.embedding
args.word2id = vocab.w2i
args.id2word = vocab.i2w
args.rl_sample_size=20
args.epsilon=0.1
args.max_num_sents=3
args.kl_method='none'
args.kl_weight=0.0095
args.model_file = "model/banditsum_kl_model.pt"

def convert_tokens_to_ids(doc, args):
    max_len = len(max(doc, key=lambda x: len(x)))
    sent_list = []
    for i in range(len(doc)):
        words = doc[i]
        sent = [args.word2id[word] if word in args.word2id else 1 for word in words]
        sent += [0 for _ in range(max_len - len(sent))]  # this is to pad at the end of each sequence
        sent_list.append(sent)
    return torch.tensor(sent_list).long()

def init_model(args):
    rewards = {"train": None, "train_single": None, "dev": None, "dev_single": None}
    model = SimpleRNN(args, rewards)
    model.cuda()
    checkpoint = torch.load(args.model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


nlp = spacy.load("en")

model = None

@app.route("/", methods=["POST"])
def summarize():
    
    global model
    
    if model is None:
        model = init_model(args)
    
    doc = nlp(request.json['text'])
    
    sentwords = []

    for sent in doc.sents:
        words = [word.text for word in sent if not word.is_punct]
        if len(words) > 1:
            sentwords.append(words)
            
    doc_ids = convert_tokens_to_ids(sentwords, args)
        
    with torch.no_grad():
            summary_idx = model(doc_ids.cuda())
        
    sents = [sent for i,sent in enumerate(doc.sents) if i in summary_idx]
    
    summ = " ".join([s.text for s in sents])
    
    return jsonify({"summary": summ})

if __name__ == "__main__":
    app.run(port=8000)