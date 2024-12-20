from flask import Flask, request, jsonify
import torch
from torch.nn import Transformer
import pickle
import pandas as pd
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from typing import Iterable, List
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import warnings
import os
import gdown
warnings.filterwarnings('ignore')

# Load các thành phần cần thiết (tokenizer, vocab, model)
SRC_LANGUAGE = 'English'
TGT_LANGUAGE = 'Vietnamese'
# Cập nhật biến SRC và TGT
SRC_LANGUAGE_VI_EN = 'Vietnamese'
TGT_LANGUAGE_VI_EN = 'English'

# Tải model từ Google Drive nếu chưa có
def download_model_from_drive(file_id, destination):
    if not os.path.exists(destination):
        print(f"Đang tải model từ Google Drive tới {destination}...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)
        print("Tải model thành công!")
    else:
        print(f"Model {destination} đã tồn tại.")

# File ID trên Google Drive
EN_VI_MODEL_ID = "1-4WnCCAHOLqZa2ROBUUNpZFUaooSHtjQ"  # Thay thế bằng ID của model enVi_transformer.pth
VI_EN_MODEL_ID = "1-AU1pgS40Tjar9o3BrW0vjvvdqLVsje7B"  # Thay thế bằng ID của model viEn_transformer.pth
TOKEN_VOCAB_ENVI_ID = "17yHIZP32t4-nS1tPIYPHj3wnzRqSmLKU"
TOKEN_VOCAB_VIEN_ID = "1qlLoYncJEfb8xWM-VBxVLimrB8EfnjcB"

# Đường dẫn lưu trữ
MODEL_PATH = './models/enVi_transformer.pth'
MODEL_PATH_VI_EN = './models/viEn_transformer.pth'
TOKEN_VOCAB_PATH = './models/token_vocab_enVi_data.pkl'
TOKEN_VOCAB_PATH_VI_EN = './models/token_vocab_viEn_data.pkl'

# Tải model từ Google Drive
os.makedirs('./models', exist_ok=True)
download_model_from_drive(EN_VI_MODEL_ID, MODEL_PATH)
download_model_from_drive(VI_EN_MODEL_ID, MODEL_PATH_VI_EN)
download_model_from_drive(TOKEN_VOCAB_ENVI_ID, TOKEN_VOCAB_PATH)
download_model_from_drive(TOKEN_VOCAB_VIEN_ID, TOKEN_VOCAB_PATH_VI_EN)


def vi_tokenizer(sentence):
    tokens = word_tokenize(sentence)
    return tokens

# Place-holders
token_transform = {}
vocab_transform = {}
# Load tokenizer và vocab cho tiếng Việt -> tiếng Anh
token_transform_vi_en = {}
vocab_transform_vi_en = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('basic_english')
token_transform[TGT_LANGUAGE] = get_tokenizer(vi_tokenizer)
token_transform_vi_en[SRC_LANGUAGE_VI_EN] = get_tokenizer(vi_tokenizer)
token_transform_vi_en[TGT_LANGUAGE_VI_EN] = get_tokenizer('basic_english')

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    for index,data_sample in data_iter:
        yield token_transform[language](data_sample[language])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# Load tokenizer và vocab
with open(TOKEN_VOCAB_PATH, 'rb') as f1:
    token_transform, vocab_transform = pickle.load(f1)
with open(TOKEN_VOCAB_PATH_VI_EN, 'rb') as f2:
    token_transform_vi_en, vocab_transform_vi_en = pickle.load(f2)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Check whether running on gpu or cpu

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(22)
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
# Cập nhật vocab và transformer cho tiếng Việt -> tiếng Anh
SRC_VOCAB_SIZE_VI_EN = len(vocab_transform_vi_en[SRC_LANGUAGE_VI_EN])
TGT_VOCAB_SIZE_VI_EN = len(vocab_transform_vi_en[TGT_LANGUAGE_VI_EN])
EMB_SIZE = 512
NHEAD = 8  # embed_dim must be divisible by num_heads
FFN_HID_DIM = 512
BATCH_SIZE = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROP_OUT = 0.1

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,DROP_OUT)
transformer_vi_en = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                       NHEAD, SRC_VOCAB_SIZE_VI_EN, TGT_VOCAB_SIZE_VI_EN, FFN_HID_DIM, DROP_OUT)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
transformer_vi_en = transformer_vi_en.to(DEVICE)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
    
# src and tgt language text transforms to convert raw strings into tensors indices
text_transform_vien = {}
for ln in [SRC_LANGUAGE_VI_EN, TGT_LANGUAGE_VI_EN]:
    text_transform_vien[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform_vi_en[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# Load trạng thái của mô hình từ file
state_dict = torch.load("./models/enVi_transformer.pth" ,map_location=torch.device('cpu'))
state_dict_vi_en = torch.load("./models/viEn_transformer.pth", map_location=torch.device('cpu'))

# Khôi phục trạng thái của mô hình từ state_dict
transformer.load_state_dict(state_dict)
transformer_vi_en.load_state_dict(state_dict_vi_en)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE).long()
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def translate_vien(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform_vien[SRC_LANGUAGE_VI_EN](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform_vi_en[TGT_LANGUAGE_VI_EN].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
                         
# Flask server
app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate_api():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'Câu đầu vào không hợp lệ!'}), 400

    input_sentence = data['sentence']
    translated_sentence = translate(transformer, input_sentence)  # Truyền model vào đây
    return jsonify({
        'input': input_sentence,
        'translation': translated_sentence
    })
@app.route('/translate_vi_en', methods=['POST'])
def translate_vi_en_api():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'Câu đầu vào không hợp lệ!'}), 400

    input_sentence = data['sentence']
    english_sentence = translate_vien(transformer_vi_en, input_sentence)  # Truyền mô hình dịch tiếng Việt sang tiếng Anh vào đây
    return jsonify({
        'input': input_sentence,
        'translation': english_sentence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
