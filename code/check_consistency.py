#!/usr/bin/env python3
"""Integrity checker for the replication bundle.

    python code/check_consistency.py

Core gate (always runs): re-executes the tie-corrected Kendall's W ranking
pipeline on the raw model responses and asserts that every construct result
reported in the paper reproduces from the data (age W=0.32, p=0.012, n=7;
SES W=0.10, n=10; the socioeconomic hiring count of 273 of 278 via the
released parser). Values must come from source, not from any stored draft.

If a manuscript source (paper/agreement_paper.tex) happens to be present in a
local author build tree, the checker additionally verifies that every
headline number appears verbatim in it. That half is skipped in this
data+code repository, which does not ship the LaTeX source.
"""
import json, sys, re

import os
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(ROOT, "data")
RAW   = os.path.join(ROOT, "data", "raw")
PAPER = os.path.join(ROOT, "paper")
canon = json.load(open(os.path.join(DATA, "canonical_values.json")))

# The manuscript source is not part of this replication repository. If a
# local copy of the .tex is present (e.g. an author build tree), the checker
# also verifies that every headline number appears verbatim in it; otherwise
# it runs the data-side checks only. The source recompute below is the real
# integrity gate and always runs.
_tex_path = os.path.join(PAPER, "agreement_paper.tex")
tex = open(_tex_path).read() if os.path.exists(_tex_path) else None

def rnd(x, n): return f"{x:.{n}f}"

# Each entry: (label, string-as-rendered-in-tex, [forbidden stale strings])
checks = [
    ("call_count",        "8{,}040",                 ["8,040"]),
    ("baseline W",        "$W=0.07$",                []),
    ("baseline p",        "$p=0.83$",                []),
    ("baseline null95",   "$0.23$",                  []),
    ("biaslab_dropped",   "$W=0.08$",                []),
    ("weak_only",         "$W=0.11$, $p=0.68$",      ["p=0.66", "p=0.64"]),
    ("regional US",       "$+0.03$",                 ["-0.13", "-0.134"]),
    ("regional CN",       "$+0.02$",                 ["-0.131"]),
    ("tost p",            "$p=0.044$",               ["TOST $p=0.014$"]),
    ("claude net",        "$+0.04$",                 []),
    ("MP items",          "248 items",               []),
    ("hiring items",      "38",                      []),
    ("em-dash",           "",                        ["\u2014"]),  # must never appear
]

# direction column means, rendered signed 2dp, must each be present somewhere
dir_render = {
    'biaslab': '-0.38', 'hiring': '-0.37', 'marked_personas': '0.57',
    'winobias': '0.27', 'opinionqa': '0.19', 'bold': '-0.04', 'refletters': '-0.04',
}

fails = 0

# cross-check canonical file itself is internally sane (always runs)
assert canon["call_count"] == 8040, "canonical call_count drifted"
assert round(canon["regional"]["US"] - canon["regional"]["CN"], 3) == canon["regional"]["diff"]

def _scan_tex(tex):
    """Optional: if a manuscript source is present, verify every headline
    number appears verbatim in it. Skipped in the data+code replication repo."""
    f = 0
    for label, expected, forbidden in checks:
        if expected and expected not in tex:
            print(f"[FAIL] {label}: expected {expected!r} not found"); f += 1
        for forb in forbidden:
            if forb in tex:
                disp = forb.encode('unicode_escape').decode()
                print(f"[FAIL] {label}: stale/forbidden {disp!r} present"); f += 1
    for tool, val in dir_render.items():
        if val not in tex:
            print(f"[WARN] direction {tool}: {val} not found verbatim (may be in table only)")
    _det_render = {"Marked Personas":"0.57","Hiring":"0.39","BiasLab":"0.38",
                   "WinoBias":"0.27","OpinionQA":"0.19","Reference Letters":"0.14",
                   "BOLD":"0.06","Discrim-Eval":"0.04"}
    _line84 = next((l for l in tex.splitlines() if "95\\% intervals excluding zero" in l and "Marked Personas" in l), "")
    for _tool,_v in _det_render.items():
        if f"{_tool} ${_v}$" not in _line84:
            print(f"[FAIL] detection {_tool}: expected {_tool} ${_v}$ in detection list"); f += 1
    for label, val in [
        ("SES W","0.10"),("SES p","0.82"),("SES hiring picks","273"),("SES hiring total","278"),
        ("age W","0.32"),("age p","0.012"),("age boot interval","0.07"),
        ("age boot interval hi","0.50"),("age drop-saturated p","0.054"),
        ("age drop-flat W","0.26"),("age drop-flat p","0.14"),
        ("age deleted models","GPT-5.2, Claude Sonnet 4, Gemini 3 Flash"),
        ("age usable rate","95.8"),("ses usable rate","96.7"),("same-six frontier W","0.16"),
        ("US-CN difference","+0.01"),("diff CI","[-0.07, 0.08]"),("tighter TOST","0.084"),
        ("title","Disagree on Ranking")]:
        if val not in tex:
            print(f"[FAIL] tex {label}: expected {val!r} in paper"); f += 1
    for forbidden in ["dollars of inference","fifteen dollars","twenty dollars","thirty dollars"]:
        if forbidden in tex:
            print(f"[FAIL] price text still present: {forbidden!r}"); f += 1
    if "Affiliation placeholder" in tex:
        print("[FAIL] affiliation placeholder still present"); f += 1
    if "\\date{\\today}" in tex:
        print("[FAIL] date still present"); f += 1
    return f

if tex is not None:
    fails += _scan_tex(tex)
else:
    print("[info] no manuscript source present; running data-side checks only.")

if fails:
    print(f"\n{fails} FAILURE(S) -- do not save until resolved")
    sys.exit(1)
print("Historical metadata checks pass; numerical checks still required below.")

# ============================================================
# SOURCE RECOMPUTE: rerun the canonical ranking pipeline from
# the raw score files and assert the paper's construct numbers
# reproduce. This is the anti-memory guard: values must come
# from data, not from prior drafts.
# ============================================================
def _recompute_from_source():
    try:
        import numpy as np, pandas as pd, scipy.stats as st
    except Exception as e:
        print(f"[FAIL] source recompute needs numpy/pandas/scipy: {e}"); return 1
    import os
    if not os.path.exists(os.path.join(DATA,"age_scores.json")):
        print("[FAIL] score files not found"); return 1
    DISP=['GPT-5.2','Claude Sonnet 4','Gemini 3 Flash','Llama 4 Maverick','Qwen3-235B',
          'DeepSeek V3.2','GLM-4.7','MiMo V2.5','Kimi K2','Mistral Large']
    def kendall_w(M):
        n,m=M.shape; R=np.apply_along_axis(st.rankdata,0,M); Rsum=R.sum(1)
        S=((Rsum-Rsum.mean())**2).sum(); T=0
        for j in range(m):
            _,c=np.unique(M[:,j],return_counts=True); T+=(c**3-c).sum()
        return 12*S/(m**2*(n**3-n)-m*T)
    def wp(S,drop=()):
        disc=[t for t in S.columns if S[t].std()>0.02 and t not in drop]
        M=S[disc].dropna().values; rng=np.random.default_rng(42)
        W=kendall_w(M); null=np.array([kendall_w(np.column_stack([rng.permutation(M[:,j]) for j in range(M.shape[1])])) for _ in range(20000)])
        return round(W,3), round((null>=W).mean(),3), M.shape[0]
    f=0
    age=pd.read_json(os.path.join(DATA,"age_scores.json")).reindex(DISP)
    ses=pd.read_json(os.path.join(DATA,"ses_scores.json")).reindex(DISP)
    aW,ap,an=wp(age); sW,sp,sn=wp(ses)
    for lbl,got,want in [("age W",f"{aW:.2f}","0.32"),("age p",f"{ap:.3f}","0.012"),
                         ("age n",str(an),"7"),("ses W",f"{sW:.2f}","0.10"),("ses n",str(sn),"10")]:
        if got!=want:
            print(f"[FAIL] source recompute {lbl}: got {got}, expected {want}"); f+=1
    dsW,dsp,dsn=wp(age,drop=('bbq','biasasker'))
    dfW,dfp,dfn=wp(age,drop=('biasasker','opinionqa'))
    for lbl,got,want in [("age drop-sat p",f"{dsp:.3f}","0.054"),("age drop-sat n",str(dsn),"7"),
                         ("age drop-flat W",f"{dfW:.2f}","0.26"),("age drop-flat p",f"{dfp:.2f}","0.14"),
                         ("age drop-flat n",str(dfn),"8")]:
        if got!=want:
            print(f"[FAIL] source recompute {lbl}: got {got}, expected {want}"); f+=1
    # SES hiring 273 of 278 via released parser
    import json as _j, re as _re
    raw=pd.DataFrame(_j.load(open(os.path.join(RAW,"ses_raw.json"))))
    items={it['id']:it for it in _j.load(open(os.path.join(DATA,"ses_items.json")))['hiring']}
    sc=[]
    for _,r in raw[raw.tool=='hiring'].iterrows():
        c=r['content']
        if not isinstance(c,str): continue
        ch=c.upper().strip(); pk='A' if ch.startswith('A') else ('B' if ch.startswith('B') else None)
        if pk is None: continue
        it=items[r['item']]; g=it['first_g'] if pk=='A' else it['second_g']
        sc.append(-1.0 if g!=it['stereo_dir'] else 1.0)
    wc=sum(1 for x in sc if x==-1.0); tot=len(sc)
    if not (wc==273 and tot==278):
        print(f"[FAIL] SES hiring working-class: got {wc} of {tot}, expected 273 of 278"); f+=1
    if f==0: print("Source recompute: all construct numbers reproduce from raw data.")
    return f

_src_fails = _recompute_from_source()
if _src_fails:
    print(f"\n{_src_fails} source-recompute failures.")
    sys.exit(1)
print("Historical construct checks passed. This is not validation of all paper claims.")
