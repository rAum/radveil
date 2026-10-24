# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Build site/docs.html from template/*.html, embedding the example programs
in examples/ verbatim (escaped and statically highlighted, so the page needs
no JavaScript library and loads nothing from elsewhere).

    python tools/docs_site/gen.py            # writes site/docs.html

Markers in the templates:
    [[tabs py="py/x.py" rs="rs/src/bin/x.rs" c="c/x.c" sh="sh/x.sh"]]
        one example per language, switchable; a value "note:<html>" shows
        text instead of code
    [[code lang title="…"]] … [[/code]]
        a single highlighted block

Every example is meant to run as written: see README.md for how to test
them against a real volume before publishing.
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "site", "docs.html")

KW = {
    "python": "and as assert break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(),
    "rust": "as break const continue crate else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while dyn Some None Ok Err".split(),
    "c": "auto break case char const continue default do double else enum extern float for goto if int long return short signed sizeof static struct switch typedef union unsigned void volatile while NULL nullptr using namespace template typename class public private throw try catch new delete auto bool true false".split(),
    "bash": "if then else fi for in do done while case esac function return export local".split(),
}
TYPES_C = set("int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t size_t FILE".split())
TOKEN = {
    "python": [
        ("com", r"#[^\n]*"),
        ("str", r'[rbf]?"""[\s\S]*?"""|[rbf]?\'\'\'[\s\S]*?\'\'\'|[rbf]?"(?:\\.|[^"\\\n])*"|[rbf]?\'(?:\\.|[^\'\\\n])*\''),
        ("dec", r"@[A-Za-z_]\w*"),
        ("num", r"\b\d+(?:\.\d+)?\b"),
        ("id", r"[A-Za-z_]\w*"),
    ],
    "rust": [
        ("com", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("str", r'b?"(?:\\.|[^"\\])*"'),
        ("mac", r"[a-z_]\w*!"),
        ("num", r"\b\d+(?:\.\d+)?(?:_?[a-z]\d*)?\b"),
        ("id", r"[A-Za-z_]\w*"),
    ],
    "c": [
        ("com", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("pp", r"#\s*(?:include|define|ifdef|ifndef|endif|if|else|pragma)[^\n]*"),
        ("str", r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\''),
        ("num", r"\b\d+(?:\.\d+)?[fFuUlL]*\b"),
        ("id", r"[A-Za-z_]\w*"),
    ],
    "bash": [
        ("com", r"(?:(?<=\s)|^)#[^\n]*"),
        ("str", r'"(?:\\.|[^"\\])*"|\'[^\']*\''),
        ("var", r"\$\w+|\$\{[^}]*\}"),
        ("flag", r"(?:(?<=\s)|^)--?[A-Za-z][\w-]*"),
        ("id", r"[A-Za-z_][\w.-]*"),
    ],
}


def highlight(code: str, lang: str) -> str:
    lang = {"cpp": "c", "sh": "bash"}.get(lang, lang)
    if lang not in TOKEN:
        return html.escape(code)
    rx = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN[lang]), re.M)
    out, pos = [], 0
    kws = set(KW[lang])
    for m in rx.finditer(code):
        out.append(html.escape(code[pos:m.start()]))
        kind, tok = m.lastgroup, m.group()
        cls = None
        if kind == "id":
            after = code[m.end():m.end() + 1]
            if tok in kws:
                cls = "kw"
            elif lang == "bash" and tok == "radveil":
                cls = "fn"
            elif lang == "bash":
                cls = None
            elif after == "(":
                cls = "fn"
            elif (lang == "rust" and tok[:1].isupper()) or (lang == "c" and (tok in TYPES_C or tok.startswith("Radveil"))):
                cls = "ty"
            elif lang == "python" and tok[:1].isupper() and len(tok) > 1:
                cls = "ty"
        else:
            cls = kind
        esc = html.escape(tok)
        out.append(f'<span class="t-{cls}">{esc}</span>' if cls else esc)
        pos = m.end()
    out.append(html.escape(code[pos:]))
    return "".join(out)


def codeblock(code: str, lang: str, title: str = "") -> str:
    code = code.strip("\n")
    head = f'<div class="code-title">{html.escape(title)}</div>' if title else ""
    return (f'<div class="code">{head}<button class="copy" type="button" aria-label="Copy code">Copy</button>'
            f'<pre data-lang="{lang}"><code>{highlight(code, lang)}</code></pre></div>')


LANGS = [("py", "Python", "python"), ("rs", "Rust", "rust"), ("c", "C / C++", "c"), ("sh", "CLI", "bash")]


def tabs(args: dict) -> str:
    btns, panes = [], []
    for key, label, lang in LANGS:
        if key not in args:
            continue
        val = args[key]
        if val.startswith("note:"):
            body = f'<p class="tab-note">{val[5:]}</p>'
        else:
            path = os.path.join(HERE, "examples", val)
            src = open(path, encoding="utf-8").read()
            l = "cpp" if val.endswith(".cpp") else lang
            body = codeblock(src, l, os.path.basename(val))
        btns.append(f'<button type="button" role="tab" data-lang="{key}">{label}</button>')
        panes.append(f'<div class="pane" role="tabpanel" data-lang="{key}">{body}</div>')
    return f'<div class="tabs"><div class="tablist" role="tablist">{"".join(btns)}</div>{"".join(panes)}</div>'


def render(tpl: str) -> str:
    tpl = re.sub(r"\[\[code (\w+)(?: title=\"([^\"]*)\")?\]\]\n([\s\S]*?)\n\[\[/code\]\]",
                 lambda m: codeblock(m.group(3), m.group(1), m.group(2) or ""), tpl)
    tpl = re.sub(r"\[\[tabs ([^\]]*)\]\]",
                 lambda m: tabs(dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))), tpl)
    return tpl


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob(os.path.join(HERE, "template", "*.html")))
    tpl = "".join(open(f, encoding="utf-8").read() for f in files)
    out = render(tpl)
    for marker in ("[[tabs", "[[code", "[[/code"):
        assert marker not in out, out[out.index(marker):out.index(marker) + 120]
    open(OUT, "w", encoding="utf-8").write(out)
    print("wrote", OUT, len(out) // 1024, "KiB")
