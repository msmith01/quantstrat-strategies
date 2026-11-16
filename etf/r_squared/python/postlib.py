"""Building blocks for the article HTML (same markup as the 4.1 and 4.2 posts): MathML equations, highlighted code boxes
with Python/R tabs. Copied from etf/alpha_rotation/python/build_post.py so that this folder is self-contained."""
import html

from pygments import highlight
from pygments.formatters import HtmlFormatter


def mi(x): return f"<mi>{x}</mi>"
def mo(x): return f"<mo>{x}</mo>"
def mn(x): return f"<mn>{x}</mn>"
def row(*a): return "<mrow>" + "".join(a) + "</mrow>"
def sub(b, s): return f"<msub>{mi(b) if len(b) == 1 else b}{mi(s) if isinstance(s, str) and len(s) == 1 else s}</msub>"
def sup(b, s): return f"<msup>{b}{s}</msup>"
def mop(x): return f'<mo stretchy="false" lspace="0" rspace="0">{x}</mo>'
def fn(name, arg): return row(name, mop("("), arg, mop(")"))
def frac(a, b): return f"<mfrac>{a}{b}</mfrac>"
def sqrt(a): return f"<msqrt>{a}</msqrt>"


def eq(n, body, label):
    return (f'<div class="qs-eq" role="img" aria-label="{html.escape(label)}"><math display="block">{body}</math>'
            f'<span class="qs-eqno">({n})</span></div>')


def hl(code, lexer):
    return highlight(code, lexer, HtmlFormatter(nowrap=True)).rstrip("\n")


def code_box(code_html, lang, fname=None, label=None):
    nm = f'<span class="qs-fname">{fname}</span>' if fname else ""
    return (f'<div class="qs-code" data-lang="{lang}"><div class="qs-code-bar"><span class="qs-lang">{label or lang}</span>{nm}</div>'
            f'<pre><code>{code_html}</code></pre></div>')


def tabs(panels, extra=""):
    inner = "".join(f'<div class="qs-tab-panel" data-tab="{n}">{h}</div>' for n, h in panels)
    return f'<div class="qs-tabs {extra}">{inner}</div>'


def hl_function(src, start, end, lexer, tag):
    """Whole file, exact text preserved, with lines start..end (1-based) in a shaded band."""
    lexer.stripnl = False
    lexer.ensurenl = False
    lines = src.split("\n")
    raw = lambda txt: highlight(txt, lexer, HtmlFormatter(nowrap=True))
    before = "\n".join(lines[:start - 1]) + "\n"
    core = "\n".join(lines[start - 1:end]) + "\n"
    after = "\n".join(lines[end:])
    return (raw(before) + f'<span class="qs-hl-label" data-label="{tag}"></span><span class="qs-hl">{raw(core)}</span>' + raw(after))
