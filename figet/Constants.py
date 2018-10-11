
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
CHAR_VOCAB = "char"


# dataset file fields
HEAD = "mid"
RIGHT_CTX = "rCtx"
LEFT_CTX = "lCtx"
TYPE = "type"

EPS = 1e-5


CHARS = ['!', '"', '#', '$', '%', '&', "'", '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
         '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd',
         'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         '{', '}', '~', '·', 'Ì', 'Û', 'à', 'ò', 'ö', '˙', 'ِ', '’', '→', '■', '□', '●', '【', '】', 'の', '・', '一', '（',
         '）', '＊', '：', '￥']
