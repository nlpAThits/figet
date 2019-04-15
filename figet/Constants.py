
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

COARSE_FLAG = 0
FINE_FLAG = 1
UF_FLAG = 2

COARSE = {'/location', '/organization', '/other', '/person'}
FINE = {'/location/city', '/organization/sports_team', '/other/religion', '/other/currency', '/other/living_thing',
        '/person/political_figure', '/organization/sports_league', '/organization/company', '/location/country',
        '/other/food', '/other/heritage', '/organization/education', '/organization/military', '/person/athlete',
        '/organization/music', '/other/sports_and_leisure', '/organization/transit', '/organization/political_party',
        '/person/title', '/other/product', '/other/supernatural', '/other/health', '/person/legal',
        '/organization/stock_exchange', '/other/event', '/location/structure', '/person/military', '/other/award',
        '/person/religious_leader', '/other/art', '/person/coach', '/location/transit', '/location/park',
        '/other/body_part', '/location/celestial', '/other/language', '/person/doctor', '/other/internet',
        '/location/geograpy', '/other/legal', '/location/geography', '/other/scientific', '/organization/government',
        '/person/artist'}
ULTRA = {'/location/geograpy/island', '/other/event/election', '/organization/company/broadcast',
         '/other/event/holiday', '/location/transit/railway', '/other/art/film', '/location/structure/hotel',
         '/person/artist/actor', '/location/structure/airport', '/other/product/car', '/other/art/music',
         '/person/artist/director', '/person/artist/author', '/other/product/mobile_phone',
         '/other/event/violent_conflict', '/other/art/writing', '/location/transit/road',
         '/location/structure/sports_facility', '/other/health/malady', '/location/geography/mountain',
         '/location/transit/bridge', '/other/living_thing/animal', '/location/geography/island',
         '/other/event/protest', '/location/structure/hospital', '/person/artist/music',
         '/other/event/natural_disaster', '/other/event/sports_event', '/location/structure/restaurant',
         '/location/geography/body_of_water', '/other/art/broadcast', '/other/product/computer',
         '/other/language/programming_language', '/other/product/weapon', '/other/health/treatment',
         '/other/art/stage', '/location/structure/theater', '/other/event/accident', '/location/structure/government',
         '/organization/company/news', '/other/product/software'}


CHARS = ['!', '"', '#', '$', '%', '&', "'", '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
         '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd',
         'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         '{', '}', '~', '·', 'Ì', 'Û', 'à', 'ò', 'ö', '˙', 'ِ', '’', '→', '■', '□', '●', '【', '】', 'の', '・', '一', '（',
         '）', '＊', '：', '￥']
