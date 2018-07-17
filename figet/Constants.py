
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = "<blank>"
UNK_WORD = "unk"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

BUFFER_SIZE = 64 * (1024 ** 2)

TOKEN_VOCAB = "token"
TYPE_VOCAB = "type"


# dataset file fields
HEAD = "mid"
RIGHT_CTX = "rCtx"
LEFT_CTX = "lCtx"
TYPE = "type"
