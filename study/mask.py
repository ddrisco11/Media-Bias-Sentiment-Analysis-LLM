"""Blind judges to the source: remove links and replace outlet names, handles and self-references with [OUTLET].

Generated tweets often reveal their voice ("Tune in to Fox News", "#FoxNews", "Read more on abc news", "our team"),
and real tweets sometimes do too. Masking applies identically to real and generated text.
"""
import re

from study.common import scoring_text

OUTLET_PATTERNS = [
    r'@?nytimes\b', r'@?nytopinion\b', r'\bNew York Times\b', r'\bThe Times\b', r'\bNYT\b',
    r'#?\bFox ?News\b(?: Digital| Channel)?', r'@?FoxNews\b', r'\bFox\b(?= (?:News|Business|Digital|reporting|team|poll|has|is))',
    r'@?Reuters\b', r'\bThomson Reuters\b',
    r'\bAssociated Press\b', r'@AP\b', r'\bAP(?= (?:News|reporting|is|has|analysis|explainer|VoteCast|Fact Check|team|poll|reporters?))',
    r"\bAP's\b", r'\bThe AP\b',
    r'\bABC News(?: Live)?\b', r'@ABC\b', r'\bABC(?= (?:News|reporting|team|poll|Live|is|has|anchor))', r'\babc news\b', r"\bABC's\b",
    r'\bGood Morning America\b', r'\bWorld News Tonight\b', r'\bThis Week\b', r'\bFox & Friends\b', r'\bThe Daily\b',
]
SELF_REFERENCE = [
    r'\b(?:our|my) (?:team|reporters?|newsroom|coverage|correspondents?|polling|poll|analysis|reporting|live blog|journalists?|fact[- ]checkers?|editors?)\b',
    r'\b(?:Stay|Follow along|Tune in|Watch|Read more|Read|Join us|Follow) (?:with|on|at|to) \[OUTLET\]',
]
MASK = '[OUTLET]'
_outlets = re.compile('|'.join(OUTLET_PATTERNS), re.I)
_self = re.compile('|'.join(SELF_REFERENCE), re.I)


def mask_outlets(text):
    text = scoring_text(text)
    text = _outlets.sub(MASK, text)
    text = _self.sub(lambda m: re.sub(r'\b(?:our|my)\b', "[OUTLET]'s", m.group(0), flags=re.I)
                     if m.group(0).lower().startswith(('our', 'my')) else m.group(0), text)
    return re.sub(r'(\[OUTLET\]\s*){2,}', MASK + ' ', text).strip()
