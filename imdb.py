from fastai.text import *
import html

BOS = "xbos"  # beginning of sentence tag
FLD = "xfld"  # data field tag

PATH = Path("data/aclImdb/")

# === 2 Standardize format ===

CLAS_PATH = Path("data/imdb_clas")
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH = Path("data/imdb_lm")
LM_PATH.mkdir(exist_ok=True)

CLASSES = ["neg", "pos", "unsup"]

'''
def get_texts(path):
    texts, labels = [], []
    for idx, label in enumerate(CLASSES):
        count = 0
        for fname in (path / label).glob("*.*"):
            count += 1
            texts.append(fname.open("r").read())
            labels.append(idx)
            if count == 1000:
                break
        print("Count: ", count)
    return np.array(texts), np.array(labels)


trn_texts, trn_labels = get_texts(PATH / "train")
val_texts, val_labels = get_texts(PATH / "test")

print("Train texts length: ", len(trn_texts))
print("Val texts length: ", len(val_texts))

col_names = ["labels", "text"]

np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = trn_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]

df_trn = pd.DataFrame({"text": trn_texts, "labels": trn_labels}, columns=col_names)
df_val = pd.DataFrame({"text": val_texts, "labels": val_labels}, columns=col_names)

# Removing unsupervised data
df_trn[df_trn["labels"] != 2].to_csv(CLAS_PATH / "train.csv", header=False, index=False)
df_val.to_csv(CLAS_PATH / "test.csv", header=False, index=False)

(CLAS_PATH / "classes.txt").open("w").writelines(f"{o}\n" for o in CLASSES)

print("Classes:", (CLAS_PATH / "classes.txt").open().readlines())

trn_texts, val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([trn_texts, val_texts]), test_size=0.1
)

print("Trn texts len: ", len(trn_texts))
print("Val texts len: ", len(val_texts))

df_trn = pd.DataFrame({"text": trn_texts, "labels": [0] * len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({"text": val_texts, "labels": [0] * len(val_texts)}, columns=col_names)

df_trn.to_csv(LM_PATH / "train.csv", header=False, index=False)
df_val.to_csv(LM_PATH / "test.csv", header=False, index=False)

# === 3 Language model tokens ===

chuncksize = 24000

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls):
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
        texts = texts.apply(fixup).values.astype(str)
    else:
        labels = df.iloc[:, range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls + 1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
        texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


df_trn = pd.read_csv(LM_PATH / "train.csv", header=None, chunksize=chuncksize)
df_val = pd.read_csv(LM_PATH / "test.csv", header=None, chunksize=chuncksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

(LM_PATH / "tmp").mkdir(exist_ok=True)
np.save(LM_PATH / "tmp" / "tok_trn.npy", tok_trn)
np.save(LM_PATH / "tmp" / "tok_val.npy", tok_val)
'''
tok_trn = np.load(LM_PATH / "tmp" / "tok_trn.npy")
tok_val = np.load(LM_PATH / "tmp" / "tok_val.npy")

print(" ".join(tok_trn[0]))

'''
Some clever tricks for tokenisation. Instead of just lower casing everything there are extra tokens to mark that upper
case begins now. Same for repeated characters. 
'''

# Numeralising all the words

freq = Counter(p for o in tok_trn for p in o)
print(freq.most_common(25))

max_vocab = 60000
min_freq = 2

# itos -> int to string
# stoi -> string to int

itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
itos.insert(0, "_unk_")
itos.insert(1, "_pad_")

stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
print("Length of itos:", len(itos))


def to_numerical(_stoi, _list):
    return np.array([[_stoi[o] for o in p] for p in _list])


trn_lm = to_numerical(stoi, tok_trn)
val_lm = to_numerical(stoi, tok_val)

print("\n", " ".join(str(o) for o in trn_lm[0]))

np.save(LM_PATH / "tmp" / "trn_ids.npy", trn_lm)
np.save(LM_PATH / "tmp" / "val_ids.npy", val_lm)
pickle.dump(itos, open(LM_PATH / "tmp" / "itos.pkl", "wb"))

trn_lm = np.load(LM_PATH / "tmp" / "trn_ids.npy")
val_lm = np.load(LM_PATH / "tmp" / "val_ids.npy")
itos = pickle.load(open(LM_PATH / "tmp" / "itos.pkl", "rb"))

vs = len(itos)
print("Word size:", vs, ", trn_lm:", len(trn_lm))

# === 4 wikitext103 conversion ===

em_sz, nh, nl = 400, 1150, 3

PRE_PATH = Path("models/wt103")
PRE_LM_PATH = PRE_PATH / "fwd_wt103.h5"

# Pretty much like dictionary
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

enc_wgts = to_np(wgts["0.encoder.weight"])
row_m = enc_wgts.mean(0)

itos2 = pickle.load((PRE_PATH / "itos_wt103.pkl").open("rb"))
stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})

# Embedding matrix
new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i, w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r >= 0 else row_m

wgts["0.encoder.weight"] = T(new_w)
wgts["0.encoder_with_dropout.embed.weight"] = T(np.copy(new_w))
wgts["1.decoder.weight"] = T(np.copy(new_w))

# === 5 Language model ===

wd = 1e-7
bptt = 70  # back property through time?
bs = 52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

t = len(np.concatenate(trn_lm))
print("t: ", t, ", t//64: ", t // 64)

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)

md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7

learner = md.get_model(opt_fn, em_sz, nh, nl, dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3],
                       dropouth=drops[4])

learner.metrics = [accuracy]
learner.unfreeze()

print(learner.get_layer_groups())

learner.model.load_state_dict(wgts)
lr = 1e-3
lrs = lr

learner.fit(lrs / 2, 1, wds=wd, use_clr=(32, 2), cycle_len=1)
learner.save("lm_last_ft")
learner.load("lm_last_ft")
learner.unfreeze()

learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
learner.sched.plot()







