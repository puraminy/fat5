# + id="X99M7UWoHC9k"
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# + colab={"base_uri": "https://localhost:8080/"} id="7OnBRq8pHFDN" outputId="8d59d75b-5ecb-49dd-90d4-32bbf22551ba"
tokenizer = T5Tokenizer.from_pretrained("/drive2/pretrained/mt5/hf/mt5-base")

# + colab={"base_uri": "https://localhost:8080/"} id="HkXHkM6OHJcH" outputId="a76804fa-7f27-418b-feac-62f58ad7b932"
model = T5ForConditionalGeneration.from_pretrained('/drive2/pretrained/mt5/hf/mt5-base')

# + [markdown] id="YMItls1shI3-"
# Our tokenizer contains 250K tokens, 

# + colab={"base_uri": "https://localhost:8080/"} id="U0vhvaP8HKm8" outputId="bd4be3d7-3847-405b-f735-e2be628dd3d5"
print(tokenizer.vocab_size)


# + [markdown] id="hX8pzm4nhhMt"
# The model has 582M parameters. 

# + colab={"base_uri": "https://localhost:8080/"} id="hz6Bv4tZIsX5" outputId="79f5c4c7-01fd-465c-9d9f-75cd9538806a"
def msize(m):
    return sum(p.numel() for p in m.parameters())

original_size = msize(model)
print(msize(model))
print(msize(model.shared))
print('encoder')
print(msize(model.encoder))
print(msize(model.encoder.block))
print('decoder')
print(msize(model.decoder))
print(msize(model.decoder.block))
print(msize(model.lm_head))

# + [markdown] id="18ckhebWLLra"
# Input and output embeddings are 66% of the whole model

# + colab={"base_uri": "https://localhost:8080/"} id="hmvmyYsyHh2s" outputId="d20aef46-4a0b-4190-f018-8bd205a02e2d"
print(msize(model.shared) / msize(model))
print(msize(model.lm_head) / msize(model))

# + [markdown] id="amFXHV9OL9SU"
# # Determine the new tokens

# + [markdown] id="NfeGCTv5Vvmu"
# Take a file from https://wortschatz.uni-leipzig.de/en/download/Russian as a representation of Russian language. It contains 1M sentences. 
#
# Also take a similar representation of English, because we want our model to be bilingual, and English shares few tokens with Russian.

# + colab={"base_uri": "https://localhost:8080/"} id="WxsNhpKfME5W" outputId="d6658201-d1ea-4371-dc1f-9561cfe8ddfd"
# #!wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/rus-ru_web-public_2019_1M.tar.gz
# #!wget https://pcai056.informatik.uni-leipzig.de/downloads/corpora/fas_newscrawl_2019_1M.tar.gz
#!!wget https://pcai056.informatik.uni-leipzig.de/downloads/corpora/fas-ir_web-public_2019_1M.tar.gz
# #!tar -xsvf rus-ru_web-public_2019_1M.tar.gz
# #!tar -xsvf fas_newscrawl_2019_1M.tar.gz
# !tar -xsvf fas-ir_web-public_2019_1M.tar.gz


# + colab={"base_uri": "https://localhost:8080/"} id="XNHwPMCHiRhr" outputId="fb672f0b-49e5-4e58-ab93-a2640c320400"
# !wget http://pcai056.informatik.uni-leipzig.de/downloads/corpora/eng-com_web-public_2018_1M.tar.gz
# !tar -xsvf eng-com_web-public_2018_1M.tar.gz

# + [markdown] id="gqjTHFJIiZTk"
# Let us look at the sentences

# + colab={"base_uri": "https://localhost:8080/", "height": 223} id="IoJlXMw_M7pT" outputId="1393a16c-58b0-4cc2-982f-566804babebe"
import pandas as pd
pd.options.display.max_colwidth = 300
import csv
fname = 'fas-ir_web-public_2019_1M/fas-ir_web-public_2019_1M-sentences.txt'
df_ru = pd.read_csv(fname, sep='\t', header=None, quoting=csv.QUOTE_NONE)
df_ru.columns = ['idx', 'text']
df_ru.sample(5)

# + colab={"base_uri": "https://localhost:8080/", "height": 258} id="V-Uc7nbziyXp" outputId="f22a6661-4f66-4adf-c0e0-98c33461efa2"
fname = 'eng-com_web-public_2018_1M/eng-com_web-public_2018_1M-sentences.txt'
df_en = pd.read_csv(fname, sep='\t', header=None, quoting=csv.QUOTE_NONE)
df_en.columns = ['idx', 'text']
df_en.sample(5)

# + [markdown] id="zhkWqfdNjNww"
# Count the tokens that the current model uses for representing the sentences. 

# + colab={"base_uri": "https://localhost:8080/", "height": 116, "referenced_widgets": ["b59e32e4abe24212a56e61823557f0e6", "aa98eac72c0e451883c18f6caf85056b", "5121e8109ad7429db64f773a3f3dd44f", "a33dc475d6e841ee9ad5f1899fa07bca", "fb694bf9c6ed4419b46a04637b1d06ba", "ce6e47d14dc44fc2b845ffbd494b3d38", "1039b1c3e9aa4bb0849744640b52980f", "287b41c1711a4a6a9027bd47dd2950ce", "bd3656150eb34d75b09acc924079bb98", "eaf6c38c53c148ea98960524d762fbdb", "e836b279e5d74e73a9097c68bd944972", "4d5a02c9c8ab45808fe5b567cf895e21", "b160e1b70c35427f96149ca991cc453c", "dd5cbbaccff644cab88cda4d7f744849", "7fb25f7d26524989a3e2326f02fdba7d", "d4a6942cad144dac8fba35fe0ee0f46c"]} id="lmzSON9iM_yb" outputId="f42aa87f-98d0-4675-bb31-2f430ef7c673"
from collections import Counter
from tqdm.auto import tqdm, trange

cnt_ru = Counter()
for text in tqdm(df_ru.text):
    cnt_ru.update(tokenizer.encode(text))

cnt_en = Counter()
for text in tqdm(df_en.text):
    cnt_en.update(tokenizer.encode(text))

# + [markdown] id="sTzND5F1OkEY"
# The tokens that are ever used with Russian are 23% of the whole vocabulary. With English, it is 27%.
#
# Surprisingly, there is more than 50% overlap between the vocabularies. Perhaps, this is because in Russian texts there are occasionally English words or other words with latin alphabet. 

# + colab={"base_uri": "https://localhost:8080/"} id="M07fj3z0NWiy" outputId="e46bc8a3-284d-43f4-c2eb-ec37b520083c"
print(tokenizer.vocab_size)
print(len(cnt_ru), len(cnt_ru)/tokenizer.vocab_size)
print(len(cnt_en), len(cnt_en)/tokenizer.vocab_size)
cmt = set(cnt_ru.keys()).intersection(set(cnt_en.keys()))
ncmt = set(cnt_ru.keys()).difference(set(cnt_en.keys()))
print("not common:", len(ncmt))
common = len(cmt)
print("common:", common)
print(common, common / len(cnt_ru))
print(len(ncmt), len(ncmt) / len(cnt_ru))


# + id="tsDn3-c2Sl36" colab={"base_uri": "https://localhost:8080/"} outputId="e5f39cba-e250-4108-9a15-24d7783d314d"
print(tokenizer.convert_ids_to_tokens([k for k in ncmt]))


# + [markdown] id="2ULUmyllmNA0"
# For both English and Russian, 10K tokens cover about 95% of the vocabulary, and 20K - about 99%. 

# + [markdown] id="kXYJn1dkcvfr"
#

# + colab={"base_uri": "https://localhost:8080/"} id="kNudkAe5NbKT" outputId="8698ba0b-9f79-4b7d-af6a-da8120e40a2a"
print('ru')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_ru.most_common(top)) / sum(cnt_ru.values()))
print('en')
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_en.most_common(top)) / sum(cnt_en.values()))

# + [markdown] id="0N_D37J3lbqr"
# Remember the old vocabulary, because we are going to replace it soon!

# + id="9RzGibfZQbgP"
old_voc = tokenizer.get_vocab()
old_inv_voc = {v: k for k, v in old_voc.items()}

# + [markdown] id="rKwEQtbRljiC"
# Look at the most used tokens. They are mostly service words or prefixes.

# + colab={"base_uri": "https://localhost:8080/"} id="Y8oL4rL8QZ8f" outputId="e777843f-4d85-4a65-b921-8d9fed12794f"
print(tokenizer.convert_ids_to_tokens([k for k, v in cnt_ru.most_common(30)]))
print(tokenizer.convert_ids_to_tokens([k for k, v in cnt_en.most_common(30)]))

# + [markdown] id="AwwPWiO3Po1x"
# We try the following composition of vocabulary:
# * 1K of top tokens of the original tokenizer (just in case)
# * Top 10K of the English vocabulary
# * Top 20K of the Russian vocabulary (or more, to make the total number of tokens 30K)
# * 100 special tokens that T5 uses
#

# + colab={"base_uri": "https://localhost:8080/"} id="J-aSMIB1Pxvh" outputId="782cf454-803f-43c6-f1fe-1d9ef0abad25"
new_tokens = set(range(1000))
for i, (k, v) in enumerate(cnt_en.most_common(10_000)):
    if k not in new_tokens:
        new_tokens.add(k)
for i, (k, v) in enumerate(cnt_ru.most_common(25_000)):
    if len(new_tokens) == 29_900:
        print(i, 'Russan tokens are included')
        break
    if k not in new_tokens:
        new_tokens.add(k)

for t in range(tokenizer.vocab_size - 100, tokenizer.vocab_size):
    new_tokens.add(t)

print(len(new_tokens))
kept_ids = sorted(new_tokens)

# + [markdown] id="BLAFLhrDoD4U"
# The new vocabulary is only 12% of the original one. 

# + colab={"base_uri": "https://localhost:8080/"} id="q21bC7tpTyuW" outputId="f57c182e-f65b-4a7b-d4d8-56445570c341"
len(kept_ids) / tokenizer.vocab_size

# + [markdown] id="s9ZrtTdcRfN_"
# The plot shows that the tokens that were more frequent in the original vocabulary more frequently get into the new vocabulary (so that the curve bends upward). 

# + colab={"base_uri": "https://localhost:8080/", "height": 279} id="IAPmeDZmRDIf" outputId="db2ac62e-b4e5-44b5-83ba-414b7efcf563"
import matplotlib.pyplot as plt
plt.plot(kept_ids)
plt.xlabel('new id of token')
plt.ylabel('old id of token');

# + [markdown] id="IaaCyAPlomLt"
# ### Update the embeddings

# + id="k-BNn3R6R0lY"
import torch

# + id="P5033SckRzzo"
new_size = len(kept_ids)
new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)

# + id="CjD6LS_9fe_M"
for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
    new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]

# + id="vv7IuBORRseE"
model.shared.weight = new_emb.weight
model.lm_head.weight = new_head.weight

# + [markdown] id="QcIDtmymo56s"
# The new model has 244M parameters - 42% of the original size. 

# + colab={"base_uri": "https://localhost:8080/"} id="g_aPBQ20kvCB" outputId="0e55c836-12ef-4392-d967-33dc466078d8"
print(msize(model), msize(model) / original_size)

# + [markdown] id="vdKmFJY_k7xZ"
# ### Update the tokenizer

# + [markdown] id="-X25sG0jmc83"
# T5 uses Sentencepiece tokenizer, which is implemented in C and is opaque to Python. 
#
# Fortunately, we can download its model and deploy it into Python using its Protobuf representation. 
#
# https://github.com/google/sentencepiece/issues/121

# + colab={"base_uri": "https://localhost:8080/"} id="OpII_eX3mY80" outputId="9e40ca83-5740-4063-e78e-e5860590ad58"
# !wget https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto

# + [markdown] id="SGb1DiYmpnkr"
# We compile the protobuf description of the sentencepiece model in order to be able to modify it. 

# + id="I6B0MA5DmaZM"
# ! protoc --python_out=. sentencepiece_model.proto

# + [markdown] id="nJwHRRzbngJY"
# Now we can serialize the model used by the current tokenizer and open it as a protobuf class. 

# + colab={"base_uri": "https://localhost:8080/", "height": 118, "referenced_widgets": ["cd4ea49927e54440b4894fd4acb6a10e", "5febfb32ad294282960c9cfec0bea3fa", "17ed0684ac8a44a5a0e0e7f910213877", "81ef6e5c3e6f43fb8b3ff8f6d4b51dcf", "fac3246b491747d2846c1e58ee5d5d2a", "92d7579873a2402d8ea3240a5f4376e4", "6dfb9542475248feae4aaaf131de16ec", "53d4aa06a7ec4040bbd4986577f786d7"]} id="MdQM0L3lnybA" outputId="4dd8acab-b095-4869-f190-86344e4d2de3"
import sentencepiece_model_pb2 as spmp
smp = tokenizer.sp_model.serialized_model_proto()
m = spmp.ModelProto()
m.ParseFromString(smp)

print('the loaded model has pieces:', len(m.pieces))
new_pieces = [m.pieces[idx] for idx in kept_ids]
print('the new pieces:', len(new_pieces))

# replace the content of the first 30K pieces
for i, p in enumerate(new_pieces):
    m.pieces[i].piece = p.piece
    m.pieces[i].score = p.score
    m.pieces[i].type = p.type

# drop the remaining pieces
n = len(new_pieces)
for i in trange(len(m.pieces) - n):
    m.pieces.pop(len(m.pieces) - 1)

print(len(m.pieces))
with open('new_sp.model', 'wb') as f:
    f.write(m.SerializeToString())

# + id="qWeP6N1sry93"
new_tokenizer = T5Tokenizer('new_sp.model', extra_ids=0)

# + [markdown] id="czfXG1IqsDT4"
# ### Save the model

# + colab={"base_uri": "https://localhost:8080/"} id="oanCNPiIsCdU" outputId="d11d9ff2-f90f-4044-99d2-d81893bdc146"
model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'cointegrated/fat5-base'
model.config

# + id="UaebisNqr4Mk"
new_tokenizer.save_pretrained('fat5-base')
model.save_pretrained('fat5-base')

# + colab={"base_uri": "https://localhost:8080/"} id="nIoB98_9r7VU" outputId="ed7c5c9a-a5b2-4738-89af-b3cc1ee97741"
# !ls fat5-base -alsh

# + [markdown] id="5gFLD5dUs7gZ"
# Now try to load the model

# + id="ewebox5usyq9"
