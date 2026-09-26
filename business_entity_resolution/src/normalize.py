"""Text normalisation for business names and addresses.

Everything here is country-agnostic string processing: transliteration of
non-Latin scripts (a learned token dictionary with a unidecode fallback),
accent folding, abbreviation canonicalisation, number clean-up and a
consonant-skeleton phonetic key.  No external data is consulted.
"""
import re

from unidecode import unidecode

# --------------------------------------------------------------------------
# Canonical forms.  Both sides of a comparison go through the same maps, so
# the canonical token only needs to be consistent, not pretty.
# --------------------------------------------------------------------------
LEGAL = {
    'inc': 'inc', 'incorporated': 'inc', 'incorporation': 'inc',
    'corp': 'corp', 'corporation': 'corp', 'co': 'co', 'company': 'co', 'cos': 'co',
    'llc': 'llc', 'ltd': 'ltd', 'limited': 'ltd', 'ltda': 'ltd',
    'pvt': 'pvt', 'private': 'pvt', 'pvtltd': 'pvt ltd', 'pte': 'pvt',
    'llp': 'llp', 'lp': 'lp', 'pllc': 'pllc', 'pc': 'pc', 'plc': 'plc', 'pa': 'pa',
    'public': 'public', 'opc': 'opc',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'sa': 'sa', 'eurl': 'eurl',
    'sci': 'sci', 'snc': 'snc', 'ei': 'ei', 'earl': 'earl', 'gaec': 'gaec', 'scp': 'scp',
    'selas': 'selas', 'gie': 'gie', 'scop': 'scop', 'selarl': 'selarl', 'scm': 'scm',
    'gmbh': 'gmbh', 'ag': 'ag', 'bv': 'bv', 'nv': 'nv', 'ltee': 'ltd',
}
# common name-word abbreviations (both spellings map to one token)
NAME_ABBR = {'etablissements': 'ets', 'etablissement': 'ets', 'compagnie': 'cie', 'societe': 'ste',
             'international': 'intl', 'internationale': 'intl'}
TITLES = {'dr', 'mr', 'mrs', 'ms', 'smt', 'shri', 'sri', 'shree', 'm', 's', 'ms', 'the',
          'dba', 'aka', 'and', 'et', 'of', 'le', 'la', 'les', 'de', 'du', 'des', 'l', 'd',
          'www', 'com', 'net', 'org', 'in', 'fr', 'esq', 'mme', 'mlle'}

ADDR = {
    'street': 'st', 'str': 'st', 'st': 'st', 'saint': 'st', 'sainte': 'ste', 'ste': 'ste',
    'road': 'rd', 'rd': 'rd', 'avenue': 'ave', 'ave': 'ave', 'av': 'ave', 'aven': 'ave',
    'drive': 'dr', 'dr': 'dr', 'drv': 'dr', 'lane': 'ln', 'ln': 'ln', 'court': 'ct', 'ct': 'ct',
    'boulevard': 'blvd', 'blvd': 'blvd', 'bd': 'blvd', 'boul': 'blvd', 'bvd': 'blvd',
    'place': 'pl', 'pl': 'pl', 'plz': 'plaza', 'terrace': 'ter', 'ter': 'ter', 'terr': 'ter',
    'highway': 'hwy', 'hwy': 'hwy', 'parkway': 'pkwy', 'pkwy': 'pkwy', 'circle': 'cir',
    'cir': 'cir', 'square': 'sq', 'sq': 'sq', 'trail': 'trl', 'trl': 'trl', 'way': 'way',
    'mount': 'mt', 'mt': 'mt', 'mountain': 'mtn', 'mtn': 'mtn', 'fort': 'ft', 'ft': 'ft',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'n': 'n', 'e': 'e', 'w': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'suite': 'ste', 'apartment': 'apt', 'apt': 'apt', 'floor': 'fl', 'fl': 'fl', 'flr': 'fl',
    'building': 'bldg', 'bldg': 'bldg', 'unit': 'unit', 'room': 'rm', 'rm': 'rm',
    'number': 'no', 'no': 'no', 'nos': 'no', 'num': 'no', 'hno': 'no', 'h': 'no', 'dno': 'no',
    'door': 'no', 'plot': 'plot', 'near': 'nr', 'nr': 'nr', 'opp': 'opp', 'opposite': 'opp',
    'township': 'twp', 'twp': 'twp', 'city': '', 'county': 'county', 'district': 'dist',
    'dist': 'dist', 'po': 'po', 'box': 'box', 'pmb': 'box',
    'nagar': 'nagar', 'ngr': 'nagar', 'colony': 'colony', 'coly': 'colony', 'sector': 'sector',
    'sec': 'sector', 'main': 'main', 'mn': 'main', 'cross': 'cross', 'crs': 'cross',
    # French
    'rue': 'rue', 'r': 'rue', 'allee': 'allee', 'all': 'allee', 'alle': 'allee',
    'chemin': 'chemin', 'ch': 'chemin', 'chem': 'chemin', 'impasse': 'imp', 'imp': 'imp',
    'faubourg': 'fbg', 'fbg': 'fbg', 'quai': 'quai', 'cours': 'cours', 'crs': 'cours',
    'route': 'rte', 'rte': 'rte', 'passage': 'pass', 'pass': 'pass', 'residence': 'res',
    'res': 'res', 'bis': 'bis', 'ter_': 'ter', 'pas': 'pass', 'batiment': 'bldg', 'bat': 'bldg',
    'ndeg': '',
    # filler / null markers
    'null': '', 'none': '', 'na': '', 'nan': '', 'nil': '', 'unknown': '',
    'of': '', 'the': '', 'and': '', 'de': '', 'du': '', 'des': '', 'la': '', 'le': '',
    'les': '', 'd': '', 'l': '', 'at': '', 'c': '', 'o': '', 'co': '',
}

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
    'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi',
    'wyoming': 'wy', 'district of columbia': 'dc', 'puerto rico': 'pr',
}
IN_STATES = {
    'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br',
    'chhattisgarh': 'cg', 'chattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr',
    'himachal pradesh': 'hp', 'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl',
    'madhya pradesh': 'mp', 'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml',
    'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od', 'orissa': 'od', 'punjab': 'pb',
    'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'tamilnadu': 'tn',
    'telangana': 'tg', 'tripura': 'tr', 'uttar pradesh': 'up', 'uttarakhand': 'uk',
    'uttaranchal': 'uk', 'west bengal': 'wb', 'delhi': 'dl', 'new delhi': 'dl',
    'jammu and kashmir': 'jk', 'jammu kashmir': 'jk', 'ladakh': 'la', 'puducherry': 'py',
    'pondicherry': 'py', 'chandigarh': 'ch', 'andaman and nicobar islands': 'an',
    'dadra and nagar haveli': 'dn', 'daman and diu': 'dd', 'lakshadweep': 'ld',
}
# French regions, and departments mapped to their region (records mix the two)
FR_REGIONS = {
    'auvergne rhone alpes': 'fr_ara', 'bourgogne franche comte': 'fr_bfc', 'bretagne': 'fr_bre',
    'centre val de loire': 'fr_cvl', 'corse': 'fr_cor', 'grand est': 'fr_ges',
    'hauts de france': 'fr_hdf', 'ile de france': 'fr_idf', 'normandie': 'fr_nor',
    'nouvelle aquitaine': 'fr_naq', 'occitanie': 'fr_occ', 'pays de la loire': 'fr_pdl',
    'provence alpes cote dazur': 'fr_pac', 'paca': 'fr_pac',
}
_FR_DEPTS = {
    'fr_ara': 'ain|allier|ardeche|cantal|drome|isere|loire|haute loire|puy de dome|rhone|savoie|haute savoie',
    'fr_bfc': 'cote dor|doubs|jura|nievre|haute saone|saone et loire|yonne|territoire de belfort',
    'fr_bre': 'cotes darmor|finistere|ille et vilaine|morbihan',
    'fr_cvl': 'cher|eure et loir|indre|indre et loire|loir et cher|loiret',
    'fr_cor': 'corse du sud|haute corse',
    'fr_ges': 'ardennes|aube|marne|haute marne|meurthe et moselle|meuse|moselle|bas rhin|haut rhin|vosges',
    'fr_hdf': 'aisne|nord|oise|pas de calais|somme',
    'fr_idf': 'seine et marne|yvelines|essonne|hauts de seine|seine saint denis|val de marne|val doise',
    'fr_nor': 'calvados|eure|manche|orne|seine maritime',
    'fr_naq': 'charente|charente maritime|correze|creuse|dordogne|gironde|landes|lot et garonne|'
              'pyrenees atlantiques|deux sevres|vienne|haute vienne',
    'fr_occ': 'ariege|aude|aveyron|gard|haute garonne|gers|herault|lot|lozere|hautes pyrenees|'
              'pyrenees orientales|tarn|tarn et garonne',
    'fr_pdl': 'loire atlantique|maine et loire|mayenne|sarthe|vendee',
    'fr_pac': 'alpes de haute provence|hautes alpes|alpes maritimes|bouches du rhone|var|vaucluse',
}
FR_DEPTS = {}
for _code, _names in _FR_DEPTS.items():
    for _n in _names.split('|'):
        FR_DEPTS[_n] = _code
_STATE_MAP = {**US_STATES, **IN_STATES, **FR_REGIONS, **FR_DEPTS}

ORDINAL_WORDS = {
    'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5', 'sixth': '6',
    'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10', 'eleventh': '11',
    'twelfth': '12', 'thirteenth': '13', 'fourteenth': '14', 'fifteenth': '15',
    'sixteenth': '16', 'seventeenth': '17', 'eighteenth': '18', 'nineteenth': '19',
    'twentieth': '20', 'thirtieth': '30', 'fortieth': '40', 'fiftieth': '50', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'premier': '1', 'premiere': '1', 'deuxieme': '2', 'troisieme': '3',
}

_ACRONYM_DOT = re.compile(r'(?<![A-Za-z0-9])([A-Za-z])\.(?=[A-Za-z]\.?)')
_SPLIT = re.compile(r"[\s,./\\\-_()\[\]{}<>|#:;\"&+*!?@=~`^$%–—·।]+")
_APOS = re.compile(r"['’‘`´]")
_ORD = re.compile(r'\b(\d+)(st|nd|rd|th|er|e|eme|ieme)\b')
_DIGIT_ALPHA = re.compile(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)')
_NONALNUM = re.compile(r'[^a-z0-9 ]+')
_DOMAIN = re.compile(r'([a-z0-9][a-z0-9\-]*)\.(com|net|org|in|co|fr|biz|info|us)\b')
_DBA = re.compile(r'\b(d\s*\.?\s*b\s*\.?\s*a\.?|aka|a\.k\.a\.?|t/a|trading as|doing business as)\b', re.I)
_VOWELS = re.compile(r'[aeiouy]')
_REPEAT = re.compile(r'(.)\1+')

TRANSLIT: dict = {}


def set_translit(d):
    TRANSLIT.clear()
    TRANSLIT.update(d)


def is_nonlatin(tok):
    return any(ord(c) > 0x24F for c in tok)


def _fold(tok):
    """Map one raw token to lowercase ASCII (learned dictionary first)."""
    if is_nonlatin(tok):
        t = TRANSLIT.get(tok)
        if t is not None:
            return t
        return unidecode(tok).lower()
    return unidecode(tok).lower()


def raw_tokens(text):
    if not text:
        return []
    text = _APOS.sub('', text)
    text = _ACRONYM_DOT.sub(r'\1', text)
    return [t for t in _SPLIT.split(text) if t]


def fold_text(text):
    """Lowercase ASCII string with transliteration applied token-wise."""
    s = ' '.join(_fold(t) for t in raw_tokens(text))
    s = _NONALNUM.sub(' ', s)
    return ' '.join(s.split())


def _clean_numbers(s):
    s = _ORD.sub(r'\1', s)
    s = _DIGIT_ALPHA.sub(' ', s)
    out = []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip('0') or '0'
        else:
            t = ORDINAL_WORDS.get(t, t)
        out.append(t)
    return out


def skeleton(tok):
    """Consonant skeleton: robust to vowel-length transliteration noise."""
    if tok.isdigit() or len(tok) < 2:
        return tok
    tok = tok.replace('ph', 'f').replace('bh', 'b').replace('dh', 'd').replace('th', 't') \
             .replace('kh', 'k').replace('gh', 'g').replace('sh', 's').replace('ch', 'c') \
             .replace('w', 'v').replace('z', 'j').replace('q', 'k').replace('x', 'ks')
    tok = _REPEAT.sub(r'\1', tok)
    return tok[0] + _VOWELS.sub('', tok[1:])


def normalize_name(name):
    """Returns dict of name representations."""
    name = name or ''
    low = unidecode(name).lower() if not any(is_nonlatin(c) for c in name) else name
    # website form: "zanderblue.com" / "www.zanderblu.com"
    domains = _DOMAIN.findall(low.lower()) if not is_nonlatin(low) else []
    domain = domains[0][0].replace('-', '') if domains else ''
    # DBA / AKA -> alternative names
    parts = [p for p in _DBA.split(name) if p and not _DBA.fullmatch(p)]
    toks = _clean_numbers(fold_text(name))
    toks = [t for t in toks if t != 'dba' and t != 'aka']
    canon = []
    for t in toks:
        c = LEGAL.get(t)
        canon.extend(c.split() if c else [NAME_ABBR.get(t, t)])
    core = [t for t in canon if t not in TITLES and not t.isdigit()
            and t not in LEGAL.values() and t not in LEGAL]
    if domain:
        core = [t for t in core if t not in ('www', 'com', 'net', 'org')]
    if not core:
        core = [t for t in canon if not t.isdigit()] or canon
    alt = ''
    if len(parts) >= 2:
        # the part that is not the main (first) name
        alt_toks = [t for t in _clean_numbers(fold_text(parts[-1]))
                    if t not in TITLES and t not in LEGAL]
        alt = ' '.join(alt_toks)
    return {
        'n_norm': ' '.join(canon),
        'n_core': ' '.join(core),
        'n_compact': ''.join(core),
        'n_skel': ' '.join(skeleton(t) for t in core),
        'n_domain': domain,
        'n_alt': alt,
        'n_nonlatin': int(any(is_nonlatin(c) for c in name)),
    }


def normalize_address(addr, country=''):
    addr = addr or ''
    out, states = [], []
    for comp in addr.split(','):
        s = fold_text(comp)
        if not s:
            continue
        # a whole component that is a bare 2-letter code ("NY", "KA", "CO") is a state
        if len(s) == 2 and s.isalpha() and s not in ('na', 'no'):
            states.append(s)
            continue
        # a component that is exactly a state name is a state; inside a longer
        # component ("kansas city", "new delhi") the words are kept as-is
        code = _STATE_MAP.get(s)
        if code:
            states.append(code)
            continue
        for t in _clean_numbers(s):
            c = ADDR.get(t, t)
            if c:
                out.append(c)
    nums = [t for t in out if t.isdigit()]
    words = [t for t in out if not t.isdigit() and len(t) > 1]
    postal = [t for t in nums if len(t) == 6]   # Indian PIN codes
    return {
        'a_norm': ' '.join(out),
        'a_words': ' '.join(words),
        'a_nums': ' '.join(nums),
        'a_first_num': nums[0] if nums else '',
        'a_postal': postal[-1] if postal else '',
        'a_state': states[-1] if states else '',
        'a_empty': int(len(out) == 0),
    }
