"""Rebuild docs/paper/main.tex from an edited Word version of the paper, keeping the edited wording exactly.

    python scripts/docx_to_paper.py --docx docs/paper/CrowdSafe_paper.docx --template docs/paper/main.tex --out docs/paper/main.tex

The running text (abstract, index terms, section and subsection titles, paragraphs, contribution list) is taken
from the Word file word for word; only characters are mapped to LaTeX (symbols, sub/superscripts, quotes,
dashes, special characters), bracketed citation numbers become \\cite of the same reference, and "Table III",
"Fig. 2", "Section V-B" become \\ref to the same object. Preamble, title, author block, table and figure blocks
and the reference list come from the template (their content is unchanged in the Word file).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
CHARS = {"×": r"$\times$", "ρ": r"$\rho$", "Σ": r"$\sum$", "⊙": r"$\odot$", "≥": r"$\geq$", "≈": r"$\approx$",
         "±": r"$\pm$", "τ": r"$\tau$", "·": r"$\cdot$", "𝒩": r"$\mathcal{N}$", "–": "--", "—": "---", "“": "``",
         "”": "''", "‘": "`", "’": "'", "ü": r'\"u', "ä": r'\"a', "ö": r'\"o', "%": r"\%", "&": r"\&", "#": r"\#",
         "_": r"\_", "†": r"$^\dagger$", "²": r"$^2$", "³": r"$^3$", "\t": " ", "\u00a0": " "}


def blocks_from_docx(path: Path) -> list[tuple[str, object]]:
    d = docx.Document(str(path))
    out = []
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(el, d)
            if el.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
                out.append(("image", None))
            runs = [(r.text, "sub" if r.font.subscript else "sup" if r.font.superscript else "n") for r in p.runs]
            if "".join(t for t, _ in runs).strip():
                out.append(("p", runs))
        elif tag == "tbl":
            out.append(("table", Table(el, d)))
    return out


def text_to_tex(runs, cite_keys: dict[int, str], refs: dict[str, str]) -> str:
    parts = []
    for t, kind in runs:
        if kind in ("sub", "sup"):
            mark = "_" if kind == "sub" else "^"
            parts.append(f"$\\,{mark}{{{t}}}$".replace("$\\,", "$") if t.strip() else t)
            continue
        parts.append(t)
    s = "".join(parts)
    # protect math chunks made above, map characters elsewhere
    pieces = re.split(r"(\$[^$]*\$)", s)
    out = []
    for i, piece in enumerate(pieces):
        if i % 2 == 1:
            out.append(piece)
            continue
        piece = piece.replace("persons/m²", "\x00PMSQ\x00").replace("m²", "m$^2$")
        piece = "".join(CHARS.get(c, c) for c in piece)
        out.append(piece)
    s = "".join(out)
    # merge "letter$_{x}$" stays as is (renders the same); clean empty math
    s = s.replace("$$", "")

    def cite(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        return "~\\cite{" + ",".join(cite_keys[n] for n in nums) + "}"
    # [1, 2] / [3, 4, 5] style lists and single [n]
    s = re.sub(r"\s?\[(\d+(?:\s*,\s*\d+)*)\]", cite, s)
    for txt, lab in sorted(refs.items(), key=lambda kv: -len(kv[0])):
        kind, num = txt.split(" ", 1)
        s = re.sub(rf"\b{re.escape(kind)} {re.escape(num)}\b(?![-\w])", f"{kind}~\\\\ref{{{lab}}}", s)
    s = s.replace("\x00PMSQ\x00", r"\pmsq{}")
    s = re.sub(r"\\pmsq\{\}(?=[.,;:)\s]|$)", r"\\pmsq{}", s)
    return re.sub(r" {2,}", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docx", type=Path, default=Path("docs/paper/CrowdSafe_paper.docx"))
    ap.add_argument("--template", type=Path, default=Path("docs/paper/main.tex"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/main.tex"))
    a = ap.parse_args()
    tpl = a.template.read_text(encoding="utf-8")

    bib = tpl[tpl.index(r"\begin{thebibliography}"):tpl.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")]
    keys = re.findall(r"\\bibitem\{([^}]+)\}", bib)
    cite_keys = {i + 1: k for i, k in enumerate(keys)}
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tpl, re.S)
    figures = re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tpl, re.S)
    tab_labels = [re.search(r"\\CStabcap\{[^}]*\}\{([^}]+)\}", t).group(1) for t in tables]
    fig_labels = [re.search(r"\\CSfigcap\{.*\}\{([^}]+)\}", f).group(1) for f in figures]
    refs = {f"Table {ROMAN[i]}": lab for i, lab in enumerate(tab_labels)}
    refs.update({f"Fig. {i + 1}": lab for i, lab in enumerate(fig_labels)})
    # section labels from the template: label -> "V-B"
    sec_labels, sec, sub, last = {}, 0, 0, ""
    for m in re.finditer(r"\\(section|subsection)\{|\\label\{(sec:[^}]+)\}", tpl):
        if m.group(1) == "section":
            sec, sub, last = sec + 1, 0, ROMAN[sec]
        elif m.group(1) == "subsection":
            sub += 1
            last = f"{ROMAN[sec - 1]}-{chr(64 + sub)}"
        else:
            sec_labels[m.group(2)] = last
    refs.update({f"Section {v}": k for k, v in sec_labels.items()})
    # which template subsection/section carries which label (by position in the template)
    label_after = {}
    for m in re.finditer(r"\\(section|subsection)\{((?:[^{}]|\{[^{}]*\})*)\}\s*\n\\label\{([^}]+)\}", tpl):
        label_after[(m.group(1), len(label_after))] = m.group(3)
    heads_with_label = [(m.group(1), m.group(3)) for m in
                        re.finditer(r"\\(section|subsection)\{((?:[^{}]|\{[^{}]*\})*)\}\s*\n\\label\{([^}]+)\}", tpl)]
    all_heads = [(m.group(1), m.start()) for m in re.finditer(r"\\(section|subsection)\{", tpl)]
    labelled_positions = {}
    for m in re.finditer(r"\\(section|subsection)\{((?:[^{}]|\{[^{}]*\})*)\}\s*\n\\label\{([^}]+)\}", tpl):
        idx = [pos for _, pos in all_heads].index(m.start())
        labelled_positions[idx] = m.group(3)

    blocks = blocks_from_docx(a.docx)
    head_end = tpl.index(r"\maketitle") + len(r"\maketitle")
    out = [tpl[:head_end], ""]
    i = 0
    # skip title/author paragraphs up to the abstract (the template keeps them)
    while not (blocks[i][0] == "p" and "".join(t for t, _ in blocks[i][1]).strip().startswith(("Abstract", "Most ", "Crowd "))
               and len("".join(t for t, _ in blocks[i][1])) > 300):
        i += 1
    abstract_runs = blocks[i][1]
    abs_text = "".join(t for t, _ in abstract_runs)
    if abs_text.startswith("Abstract—"):
        abstract_runs = [(abs_text[len("Abstract—"):], "n")]
    out += [r"\begin{abstract}", text_to_tex(abstract_runs, cite_keys, refs), r"\end{abstract}", ""]
    i += 1
    kw = "".join(t for t, _ in blocks[i][1]).replace("Index Terms—", "").strip()
    out += [r"\begin{IEEEkeywords}", kw, r"\end{IEEEkeywords}", ""]
    i += 1

    t_i = f_i = head_i = 0
    pending_image = False
    enum_open = False
    skip_notes = {"†Published results", "Horizons 10", "≈: estimated"}
    while i < len(blocks):
        kind, val = blocks[i]
        i += 1
        if kind == "image":
            pending_image = True
            continue
        if kind == "table":
            out += [tables[t_i], ""]
            t_i += 1
            continue
        text = "".join(t for t, _ in val).strip()
        if text == "REFERENCES":
            break
        if enum_open and not re.match(r"^\d+\)\s", text):
            out += [r"\end{enumerate}", ""]
            enum_open = False
        if any(text.lstrip().startswith(n) for n in skip_notes):
            continue                                          # table notes live inside the table blocks
        if re.match(r"^TABLE [IVX]+:", text):
            continue                                          # caption is part of the table block
        if text.startswith("Fig. ") and pending_image:
            out += [figures[f_i], ""]
            f_i += 1
            pending_image = False
            continue
        m = re.match(r"^([IVX]+)\.\s+(.+)$", text)
        if m and text.upper() == text:
            name = m.group(2).title().replace("And", "and").replace("Of", "of").replace("For", "for")
            out += [f"\\section{{{name}}}"]
            if head_i in labelled_positions:
                out.append(f"\\label{{{labelled_positions[head_i]}}}")
            head_i += 1
            continue
        is_sub = (len(text) < 80 and not text.endswith(".") and not re.match(r"^\d+\)", text)
                  and (re.match(r"^[A-L]\.\s", text) or text[:1].isupper()) and i < len(blocks)
                  and "TABLE" not in text and not text.startswith("Fig."))
        if is_sub:
            name = re.sub(r"^[A-L]\.\s+", "", text).strip()
            name = name.replace("persons/m2", "persons/m$^2$")
            name = re.sub(r"\bm2\b", r"\\texorpdfstring{m$^2$}{m2}", name) if "m2" in name else name
            name = name.replace("J\u00fclich", r'J\"ulich')
            out += [f"\\subsection{{{name}}}"]
            if head_i in labelled_positions:
                out.append(f"\\label{{{labelled_positions[head_i]}}}")
            head_i += 1
            continue
        m = re.match(r"^\d+\)\s+(.*)$", text)
        if m:
            if not enum_open:
                out.append(r"\begin{enumerate}")
                enum_open = True
            out.append(r"\item " + text_to_tex([(m.group(1), "n")], cite_keys, refs))
            continue
        out += [text_to_tex(val, cite_keys, refs), ""]
    if enum_open:
        out.append(r"\end{enumerate}")
    out += [bib, "", r"\end{document}", ""]
    a.out.write_text("\n".join(out), encoding="utf-8")
    print("wrote", a.out, "| tables", t_i, "| figures", f_i, "| headings", head_i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
