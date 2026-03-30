"""
report_agent/renderer.py
Rendu PDF mécanique — assemble le document à partir d'un narratif (dict texte)
et d'un payload (données).

NE RÉDIGE RIEN — injecte le texte du dict narratif aux bons endroits.
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image as RLImage, NextPageTemplate, Frame, PageTemplate, BaseDocTemplate,
)

# ─── Design system ────────────────────────────────────────────────────────────
C_PRI = HexColor("#1B3A5C")
C_SEC = HexColor("#2E6EA6")
C_ACC = HexColor("#D4A843")
C_TXT = HexColor("#2C2C2C")
C_MUT = HexColor("#6B7280")
C_THD = HexColor("#1B3A5C")
C_ALT = HexColor("#EDF2F7")
C_WARN = HexColor("#7C3800")
C_WARN_BG = HexColor("#FEF3C7")

PAGE_W, PAGE_H = A4
ML, MR, MT, MB = 25 * mm, 25 * mm, 25 * mm, 25 * mm
AW = PAGE_W - ML - MR

# ─── Configs formules par méthode ─────────────────────────────────────────────
_SMOOTHING_CFG = {
    "whittaker_henderson": {
        "titre": "Lissage par Whittaker-Henderson",
        "formules": [
            (r"$\hat{q}_x = \frac{D_x}{E_x}$", "f_qbrut.png", 18, 4, "(1)"),
            (
                r"$F(\lambda) = \sum_{x} w_x (\hat{q}_x - q_x)^2"
                r" + \lambda \sum_{x} (\Delta^2 q_x)^2$",
                "f_wh.png", 16, 12, "(2)",
            ),
            (
                r"$\mathbf{q}^* = (\mathbf{W} + \lambda \mathbf{D}^\top \mathbf{D})^{-1}"
                r" \mathbf{W} \hat{\mathbf{q}}$",
                "f_whsol.png", 16, 10, "(3)",
            ),
        ],
    },
    "gompertz_makeham": {
        "titre": "Ajustement par Gompertz-Makeham",
        "formules": [
            (r"$\hat{q}_x = \frac{D_x}{E_x}$", "f_qbrut.png", 18, 4, "(1)"),
            (r"$\mu_x = a + b \cdot c^x$", "f_gm.png", 18, 5, "(2)"),
            (r"$q_x = 1 - \exp(-\int_x^{x+1} \mu_t \, dt)$", "f_gmqx.png", 16, 8, "(3)"),
        ],
    },
    "beard": {
        "titre": "Ajustement par le modèle de Beard",
        "formules": [
            (r"$\hat{q}_x = \frac{D_x}{E_x}$", "f_qbrut.png", 18, 4, "(1)"),
            (r"$\mu_x = \frac{a + b c^x}{1 + d b c^x}$", "f_beard.png", 16, 8, "(2)"),
        ],
    },
}
_SMOOTHING_DEFAULT = {
    "titre": "Lissage actuariel",
    "formules": [(r"$\hat{q}_x = \frac{D_x}{E_x}$", "f_qbrut.png", 18, 4, "(1)")],
}

_COMMON_FORMULAS = [
    (r"$SMR = \frac{D}{E} = \frac{\sum D_i}{\sum n_i q_i}$", "f_smr.png", 16, 8),
    (
        r"$\frac{D}{E}\!\left(1-\frac{1}{9D}-\frac{u_{1-\alpha/2}}{3\sqrt{D}}\right)^{\!3}"
        r"\leq SMR \leq"
        r"\frac{D+1}{E}\!\left(1-\frac{1}{9(D+1)}+\frac{u_{1-\alpha/2}}{3\sqrt{D+1}}\right)^{\!3}$",
        "f_smric.png", 12, 14,
    ),
    (
        r"$Z = \sum n_i \frac{(\hat{q}_i - q_i)^2}{q_i(1-q_i)} \sim \chi^2(p-r-1)$",
        "f_chi2.png", 16, 12,
    ),
    (r"$\alpha_x = \frac{q_x^{exp}}{q_x^{ref}}$", "f_abat.png", 16, 5),
]


# ─── Helpers texte ────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _paras(items: list, style) -> list:
    """Convertit une liste de strings en flowables Paragraph."""
    result = []
    for item in (items or []):
        if item and item.strip():
            result.append(Paragraph(_esc(item.strip()), style))
    return result


# ─── Styles ───────────────────────────────────────────────────────────────────

def _st() -> dict:
    b = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("H1", parent=b["Heading1"], fontName="Helvetica-Bold",
                              fontSize=16, textColor=C_PRI, spaceBefore=10 * mm, spaceAfter=5 * mm),
        "h2": ParagraphStyle("H2", parent=b["Heading2"], fontName="Helvetica-Bold",
                              fontSize=12, textColor=C_SEC, spaceBefore=6 * mm, spaceAfter=3 * mm),
        "body": ParagraphStyle("B", parent=b["Normal"], fontName="Helvetica", fontSize=9.5,
                                textColor=C_TXT, leading=14, alignment=TA_JUSTIFY, spaceAfter=3 * mm),
        "cap": ParagraphStyle("C", parent=b["Normal"], fontName="Helvetica-Oblique",
                               fontSize=8, textColor=C_MUT, alignment=TA_CENTER,
                               spaceBefore=2 * mm, spaceAfter=4 * mm),
        "note": ParagraphStyle("N", parent=b["Normal"], fontName="Helvetica", fontSize=8,
                                textColor=C_MUT, leading=11, leftIndent=5 * mm,
                                spaceBefore=2 * mm, spaceAfter=2 * mm),
        "warn": ParagraphStyle("W", parent=b["Normal"], fontName="Helvetica-Bold", fontSize=9,
                                textColor=C_WARN, leading=13, leftIndent=4 * mm,
                                rightIndent=4 * mm, spaceBefore=3 * mm, spaceAfter=2 * mm,
                                backColor=C_WARN_BG, borderPadding=4),
        "ctitle": ParagraphStyle("CT", fontName="Helvetica-Bold", fontSize=28,
                                  textColor=C_PRI, alignment=TA_CENTER, spaceAfter=5 * mm),
        "csub": ParagraphStyle("CS", fontName="Helvetica", fontSize=14,
                                textColor=C_SEC, alignment=TA_CENTER, spaceAfter=3 * mm),
        "cmeta": ParagraphStyle("CM", fontName="Helvetica", fontSize=10,
                                 textColor=C_MUT, alignment=TA_CENTER, spaceAfter=2 * mm),
        "toc": ParagraphStyle("TOC", parent=b["Normal"], fontSize=11,
                               spaceBefore=3 * mm, spaceAfter=1 * mm),
    }


# ─── Table helper ─────────────────────────────────────────────────────────────

def _mt(data: list, cw: list | None = None) -> Table:
    if not cw:
        cw = [AW / len(data[0])] * len(data[0])
    t = Table(data, colWidths=cw, repeatRows=1)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), C_TXT),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#D1D5DB")),
        ("BACKGROUND", (0, 0), (-1, 0), C_THD),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), C_ALT))
    t.setStyle(TableStyle(cmds))
    return t


def _sa(ages: list) -> list:
    a_min, a_max = int(min(ages)), int(max(ages))
    step = max(1, (a_max - a_min) // 20)
    s = list(range(a_min, a_max + 1, step))
    if a_max not in s:
        s.append(a_max)
    return s


# ─── Formules LaTeX ───────────────────────────────────────────────────────────

def _rf(latex: str, fname: str, fdir: str, fs: int = 14, fw: int = 14) -> str:
    fig, ax = plt.subplots(figsize=(fw, 0.8))
    ax.text(0.02, 0.5, latex, fontsize=fs, ha="left", va="center",
            family="serif", math_fontfamily="cm")
    ax.axis("off")
    p = os.path.join(fdir, fname)
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return p


def _fb(path: str, label: str = "") -> list:
    items: list = [Spacer(1, 2 * mm)]
    img = Image.open(path)
    asp = img.size[1] / img.size[0]
    w = min(img.size[0] / 180 * 72 * 0.85, AW - 20 * mm)
    items.append(RLImage(path, width=w, height=w * asp))
    if label:
        items.append(Paragraph(label, ParagraphStyle(
            "FL", fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=C_MUT, alignment=TA_RIGHT, spaceAfter=1 * mm,
        )))
    items.append(Spacer(1, 2 * mm))
    return items


# ─── Graphiques (courbes uniquement) ──────────────────────────────────────────

def _setup() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.3})


def _g_taux(p: dict, gd: str) -> str:
    db, lis = p["donnees_brutes"], p["lissage"]
    ages = np.array(db["ages"])
    exp = np.array(db["exposure"])
    fig, ax = plt.subplots(figsize=(10, 5))
    m = exp > 10
    ax.scatter(ages[m], np.array(db["q_brut"])[m] * 1000,
               s=12, alpha=0.4, color="#6B7280", label=r"Taux bruts $\hat{q}_x$ (‰)")
    ax.plot(ages, np.array(lis["q_lisse"]) * 1000,
            color="#2E6EA6", lw=2.2, label=r"Taux lissés $q_x^*$ (‰)")
    ax.plot(ages, np.array(lis["q_ref"]) * 1000,
            color="#C53030", lw=1.5, ls="--", label=r"Référence $q_x^{ref}$ (‰)")
    ax.fill_between(ages, np.array(lis["ic_inf"]) * 1000, np.array(lis["ic_sup"]) * 1000,
                    alpha=0.15, color="#2E6EA6", label="IC 95%")
    ax.set(xlabel="Âge", ylabel="Taux (‰)", title="Taux bruts et lissés par âge",
           xlim=(min(ages), max(ages)))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    pa = os.path.join(gd, "g_taux.png")
    fig.savefig(pa, dpi=180, bbox_inches="tight")
    plt.close()
    return pa


def _g_smr(p: dict, gd: str) -> str:
    dec = p["deciles"]
    xm = [(d["age_start"] + d["age_end"]) / 2 for d in dec]
    smrs = [d["smr"] for d in dec]
    yl = [s - d["smr_ic_inf"] for s, d in zip(smrs, dec)]
    yh = [d["smr_ic_sup"] - s for s, d in zip(smrs, dec)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(xm, smrs, yerr=[yl, yh], fmt="o-", color="#2E6EA6", lw=2, ms=6,
                capsize=4, ecolor="#6B7280", label="SMR (IC 95%)")
    ax.axhline(1.0, color="#C53030", ls="--", lw=1.2, label="SMR=1")
    for x, sv in zip(xm, smrs):
        ax.scatter([x], [sv], color="#C53030" if sv > 1 else "#276749",
                   s=50, zorder=5, edgecolors="white", lw=0.5)
    ax.set(xlabel="Âge moyen du décile", ylabel="SMR", title="SMR par décile (IC 95%)")
    ax.legend()
    ax.set_ylim(0, max(d["smr_ic_sup"] for d in dec) * 1.3)
    for x, d in zip(xm, dec):
        ax.annotate(d["tranche_label"], (x, 0.02),
                    fontsize=7, ha="center", color="#6B7280", rotation=45)
    pa = os.path.join(gd, "g_smr.png")
    fig.savefig(pa, dpi=180, bbox_inches="tight")
    plt.close()
    return pa


def _g_deces(p: dict, gd: str) -> str:
    db, lis = p["donnees_brutes"], p["lissage"]
    ages = np.array(db["ages"])
    d_obs = np.array(db["deaths_observed"])
    d_exp = np.array(db["exposure"]) * np.array(lis["q_ref"])
    bands = np.arange(int(min(ages)), int(max(ages)) + 1, 5)
    obs_b, exp_b, mids = [], [], []
    for i in range(len(bands) - 1):
        m = (ages >= bands[i]) & (ages < bands[i + 1])
        obs_b.append(d_obs[m].sum())
        exp_b.append(d_exp[m].sum())
        mids.append((bands[i] + bands[i + 1]) / 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mids, obs_b, "o-", color="#2E6EA6", lw=2, ms=5, label="Observés")
    ax.plot(mids, exp_b, "s--", color="#C53030", lw=1.5, ms=5, label="Attendus")
    ax.set(xlabel="Âge (quinquennal)", ylabel="Décès",
           title="Décès observés vs attendus")
    ax.legend()
    ax.set_ylim(bottom=0)
    pa = os.path.join(gd, "g_deces.png")
    fig.savefig(pa, dpi=180, bbox_inches="tight")
    plt.close()
    return pa


def _g_abat(p: dict, gd: str) -> str:
    ab = p["abattement"]
    ages = np.array(ab["ages"])
    alpha = np.array(ab["alpha"])
    qr = np.array(ab["q_reference"])
    ag = p["validation"]["abattement_global"]
    m = qr > 0.0005
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ages[m], alpha[m], color="#2E6EA6", lw=2, label=r"$\alpha_x$")
    ax.axhline(1.0, color="#C53030", ls="--", lw=1.2, label="Ratio=1")
    ax.axhline(ag, color="#D4A843", ls=":", lw=1.5, label=f"Global={ag:.3f}")
    ax.fill_between(ages[m], alpha[m], 1.0, where=alpha[m] < 1, alpha=0.1, color="#276749")
    ax.fill_between(ages[m], alpha[m], 1.0, where=alpha[m] >= 1, alpha=0.1, color="#C53030")
    ax.set(xlabel="Âge", ylabel="Abattement", title="Abattement par âge")
    ax.legend(loc="upper right")
    pa = os.path.join(gd, "g_abat.png")
    fig.savefig(pa, dpi=180, bbox_inches="tight")
    plt.close()
    return pa


def _g_expo(p: dict, gd: str) -> str:
    db = p["donnees_brutes"]
    ages = np.array(db["ages"])
    exp = np.array(db["exposure"])
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(ages, 0, exp, alpha=0.3, color="#2E6EA6")
    ax.plot(ages, exp, color="#2E6EA6", lw=1.5)
    for d in p["deciles"]:
        ax.axvline(d["age_start"], color="#D4A843", ls=":", lw=0.8, alpha=0.7)
    ax.set(xlabel="Âge", ylabel="Exposition",
           title="Exposition par âge (frontières déciles)")
    ax.set_xlim(min(ages), max(ages))
    ax.set_ylim(bottom=0)
    pa = os.path.join(gd, "g_expo.png")
    fig.savefig(pa, dpi=180, bbox_inches="tight")
    plt.close()
    return pa


# ─── Document template ────────────────────────────────────────────────────────

class _RT(BaseDocTemplate):
    def __init__(self, fn: str, **kw):
        super().__init__(fn, **kw)
        self.pc = 0
        fb = Frame(ML, MB + 10 * mm, AW, PAGE_H - MT - MB - 15 * mm, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover",
                         frames=[Frame(0, 0, PAGE_W, PAGE_H, id="c")],
                         onPage=lambda c, d: None),
            PageTemplate(id="main", frames=[fb], onPage=self._dp),
        ])

    def _dp(self, cv, doc):
        self.pc += 1
        cv.saveState()
        yh = PAGE_H - 18 * mm
        cv.setStrokeColor(C_ACC)
        cv.setLineWidth(0.8)
        cv.line(ML, yh, PAGE_W - MR, yh)
        cv.setFont("Helvetica", 7)
        cv.setFillColor(C_MUT)
        cv.drawString(ML, yh + 2 * mm, "SIMULATION")
        cv.drawRightString(PAGE_W - MR, yh + 2 * mm, "CONFIDENTIEL")
        yf = MB
        cv.line(ML, yf, PAGE_W - MR, yf)
        cv.drawString(ML, yf - 4 * mm, "Agent actuariel — Mortality")
        cv.drawRightString(PAGE_W - MR, yf - 4 * mm, f"Page {self.pc}")
        cv.restoreState()


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def render_pdf(narratif: dict, payload: dict, output_path: str) -> str:
    """Assemble le PDF à partir du narratif (textes rédigés) et du payload (données).

    Ne rédige aucun texte — injecte le contenu du dict narratif tel quel.
    """
    wd = tempfile.mkdtemp(prefix="renderer_")
    fd = os.path.join(wd, "formulas")
    gd = os.path.join(wd, "graphs")
    os.makedirs(fd)
    os.makedirs(gd)
    _setup()

    pf = payload["portfolio"]
    db = payload["donnees_brutes"]
    lis = payload["lissage"]
    val = payload["validation"]
    dec = payload["deciles"]
    abat = payload["abattement"]
    qd = payload["qualite_donnees"]
    tr = payload.get("trace")

    ages = np.array(db["ages"])
    exp = np.array(db["exposure"])
    methode = lis["methode"]
    params = lis["parametres"]
    mcfg = _SMOOTHING_CFG.get(methode, _SMOOTHING_DEFAULT)

    # Rendre les formules LaTeX
    rend: dict[str, str] = {}
    for latex, fn, fs, fw, _ in mcfg["formules"]:
        rend[fn] = _rf(latex, fn, fd, fs, fw)
    for item in _COMMON_FORMULAS:
        latex, fn, fs, fw = item
        rend[fn] = _rf(latex, fn, fd, fs, fw)

    # Générer les graphiques
    gt = _g_taux(payload, gd)
    gs = _g_smr(payload, gd)
    gdc = _g_deces(payload, gd)
    ga = _g_abat(payload, gd)
    ge = _g_expo(payload, gd)

    s = _st()
    story: list = []
    total_d = int(np.sum(db["deaths_observed"]))
    total_e = int(np.sum(exp))
    tl = pf.get("type_contrat", "").replace("_", " ").title()
    sample = _sa(ages.tolist())
    a0 = int(ages[0])

    q_brut_arr = np.array(db["q_brut"])
    q_lisse_arr = np.array(lis["q_lisse"])
    q_ref_arr = np.array(lis["q_ref"])
    ic_i_arr = np.array(lis["ic_inf"])
    ic_s_arr = np.array(lis["ic_sup"])
    d_arr = np.array(db["deaths_observed"])
    al_arr = np.array(abat["alpha"])
    qc_arr = np.array(abat["q_construit"])
    qr2_arr = np.array(abat["q_reference"])

    # Shortcuts pour les sections narratif
    def _sec(key: str) -> dict:
        return narratif.get(key) or {}

    # ── Couverture ─────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 60 * mm),
        Paragraph("Certification de la table de<br/>mortalité d'expérience", s["ctitle"]),
        Spacer(1, 5 * mm),
        Paragraph(f"Contrats {_esc(tl)} — Étude actuarielle", s["csub"]),
        Spacer(1, 15 * mm),
        Paragraph("SIMULATION", s["csub"]),
        Spacer(1, 10 * mm),
        Paragraph(f"Période : {_esc(pf['periode_debut'])} → {_esc(pf['periode_fin'])}",
                  s["cmeta"]),
        Paragraph(f"Segmentation : {_esc(pf['segmentation'].title())}", s["cmeta"]),
        Spacer(1, 20 * mm),
        Paragraph("Rapport généré par l'agent actuariel IA", s["cmeta"]),
        NextPageTemplate("main"),
        PageBreak(),
    ]

    # ── Préambule ──────────────────────────────────────────────────────────────
    story += [Paragraph("PRÉAMBULE", s["h1"])]
    preambule = narratif.get("preambule", "")
    if preambule:
        story.append(Paragraph(_esc(preambule), s["body"]))
    story.append(PageBreak())

    # ── Sommaire ───────────────────────────────────────────────────────────────
    story.append(Paragraph("SOMMAIRE", s["h1"]))
    for sec in [
        "1. Les contrats",
        "2. Les données transmises",
        "3. Méthodologie actuarielle",
        "4. Construction de la table",
        "5. Commentaires",
        "6. Conclusion et recommandations",
        "Annexe — Paramètres et traçabilité",
    ]:
        story.append(Paragraph(sec, s["toc"]))
    story.append(PageBreak())

    # ── Section 1 — Les contrats ───────────────────────────────────────────────
    story.append(Paragraph("1. LES CONTRATS", s["h1"]))
    story += _paras(_sec("section_1_contrats").get("paragraphes"), s["body"])
    story += [
        Paragraph("<b>Tableau 1</b> — Portefeuille", s["cap"]),
        _mt(
            [
                ["Indicateur", "Valeur"],
                ["Assurés", f"{pf['n_assures']:,}"],
                ["Contrats actifs", f"{pf['n_contrats_actifs']:,}"],
                ["Période", f"{pf['periode_debut']} – {pf['periode_fin']}"],
                ["Âges", f"{pf['age_min']}–{pf['age_max']}"],
                ["Décès", f"{total_d:,}"],
                ["Exposition", f"{total_e:,} AP"],
                ["Référence", pf["table_reference"]],
            ],
            [AW * 0.5, AW * 0.5],
        ),
        PageBreak(),
    ]

    # ── Section 2 — Données ────────────────────────────────────────────────────
    sec2 = _sec("section_2_donnees")
    story.append(Paragraph("2. LES DONNÉES TRANSMISES", s["h1"]))
    story += _paras(sec2.get("paragraphes_avant_tableaux"), s["body"])
    story += [
        Paragraph("<b>Tableau 2</b> — Traitements appliqués", s["cap"]),
        _mt(
            [["Traitement", "Description"]]
            + [[t["nom"], t["description"]] for t in qd["traitements_appliques"]],
            [AW * 0.35, AW * 0.65],
        ),
        Spacer(1, 4 * mm),
        Paragraph("<b>Tableau 3</b> — Statistiques annuelles", s["cap"]),
        _mt(
            [["Année", "Exposition", "Âge moyen", "Décès"]]
            + [
                [str(y["annee"]), f"{y['exposition']:,}",
                 f"{y['age_moyen']:.1f}", str(y["deces"])]
                for y in qd["stats_annuelles"]
            ],
            [AW * 0.25] * 4,
        ),
        Spacer(1, 4 * mm),
        Paragraph("<b>Figure 1</b> — Exposition par âge (frontières déciles)", s["cap"]),
        RLImage(ge, width=AW, height=AW * 0.35),
    ]
    story += _paras(sec2.get("paragraphes_apres_tableaux"), s["body"])
    story.append(PageBreak())

    # ── Section 3 — Méthodologie ───────────────────────────────────────────────
    sec3 = _sec("section_3_methodologie")
    story.append(Paragraph("3. MÉTHODOLOGIE ACTUARIELLE", s["h1"]))
    story.append(Paragraph("3.1. Taux bruts", s["h2"]))
    if sec3.get("intro"):
        story.append(Paragraph(_esc(sec3["intro"]), s["body"]))

    f0 = mcfg["formules"][0]
    story += _fb(rend[f0[1]], f0[4])

    story.append(Paragraph(f"3.2. {_esc(mcfg['titre'])}", s["h2"]))
    if sec3.get("commentaire_lissage"):
        story.append(Paragraph(_esc(sec3["commentaire_lissage"]), s["body"]))
    for fi in mcfg["formules"][1:]:
        story += _fb(rend[fi[1]], fi[4])

    story.append(Paragraph("3.3. SMR", s["h2"]))
    if sec3.get("commentaire_smr"):
        story.append(Paragraph(_esc(sec3["commentaire_smr"]), s["body"]))
    story += _fb(rend["f_smr.png"], "")
    story += _fb(rend["f_smric.png"], "")

    story.append(Paragraph("3.4. Test du χ²", s["h2"]))
    if sec3.get("commentaire_chi2"):
        story.append(Paragraph(_esc(sec3["commentaire_chi2"]), s["body"]))
    story += _fb(rend["f_chi2.png"], "")

    story.append(Paragraph("3.5. Abattement", s["h2"]))
    if sec3.get("commentaire_abattement"):
        story.append(Paragraph(_esc(sec3["commentaire_abattement"]), s["body"]))
    story += _fb(rend["f_abat.png"], "")

    story.append(Paragraph("3.6. Déciles d'exposition", s["h2"]))
    if sec3.get("commentaire_deciles"):
        story.append(Paragraph(_esc(sec3["commentaire_deciles"]), s["body"]))
    story.append(PageBreak())

    # ── Section 4 — Construction ───────────────────────────────────────────────
    sec4 = _sec("section_4_construction")
    story.append(Paragraph("4. CONSTRUCTION DE LA TABLE", s["h1"]))
    if sec4.get("intro_taux_bruts"):
        story.append(Paragraph(_esc(sec4["intro_taux_bruts"]), s["body"]))

    story += [
        Paragraph("<b>Tableau 4</b> — Taux bruts (Dx : décès, Ex : exposition, q̂x = Dx/Ex)",
                  s["cap"]),
        _mt(
            [["Âge", "Dx", "Ex", "q̂x (‰)"]]
            + [
                [str(a), str(int(d_arr[a - a0])), str(int(exp[a - a0])),
                 f"{q_brut_arr[a - a0] * 1000:.2f}"]
                for a in sample
                if 0 <= a - a0 < len(ages)
            ],
            [AW * 0.15, AW * 0.22, AW * 0.28, AW * 0.35],
        ),
        Spacer(1, 4 * mm),
    ]

    if sec4.get("commentaire_taux_lisses"):
        story.append(Paragraph(_esc(sec4["commentaire_taux_lisses"]), s["body"]))

    story += [
        Paragraph("<b>Tableau 5</b> — Taux lissés vs référence", s["cap"]),
        _mt(
            [["Âge", "q*x (‰)", "qref (‰)", "IC inf", "IC sup"]]
            + [
                [str(a), f"{q_lisse_arr[a - a0] * 1000:.2f}",
                 f"{q_ref_arr[a - a0] * 1000:.2f}",
                 f"{ic_i_arr[a - a0] * 1000:.2f}", f"{ic_s_arr[a - a0] * 1000:.2f}"]
                for a in sample
                if 0 <= a - a0 < len(ages)
            ],
            [AW * 0.12] + [AW * 0.22] * 4,
        ),
        Spacer(1, 4 * mm),
    ]

    if sec4.get("commentaire_figure_taux"):
        story.append(Paragraph(_esc(sec4["commentaire_figure_taux"]), s["body"]))
    story += [
        Paragraph(
            "<b>Figure 2</b> — Taux bruts (points) + lissés (courbe) + référence (tirets) + IC",
            s["cap"]),
        RLImage(gt, width=AW, height=AW * 0.50),
        PageBreak(),
    ]

    if sec4.get("intro_abattement"):
        story.append(Paragraph(_esc(sec4["intro_abattement"]), s["body"]))

    story += [
        Paragraph("<b>Tableau 6</b> — Abattement (αx = q construit / q référence)", s["cap"]),
        _mt(
            [["Âge", "q construit (‰)", "q ref (‰)", "αx"]]
            + [
                [str(a), f"{qc_arr[a - a0] * 1000:.2f}",
                 f"{qr2_arr[a - a0] * 1000:.2f}", f"{al_arr[a - a0]:.3f}"]
                for a in sample
                if 0 <= a - a0 < len(ages) and qr2_arr[a - a0] > 0.0005
            ],
            [AW * 0.12, AW * 0.28, AW * 0.30, AW * 0.30],
        ),
        Spacer(1, 4 * mm),
    ]

    if sec4.get("commentaire_figure_abattement"):
        story.append(Paragraph(_esc(sec4["commentaire_figure_abattement"]), s["body"]))
    story += [
        Paragraph(f"<b>Figure 3</b> — Abattement (global={val['abattement_global']:.3f})",
                  s["cap"]),
        RLImage(ga, width=AW, height=AW * 0.45),
        PageBreak(),
    ]

    # ── Section 5 — Commentaires ───────────────────────────────────────────────
    sec5 = _sec("section_5_commentaires")
    story.append(Paragraph("5. COMMENTAIRES", s["h1"]))
    story += _paras(sec5.get("paragraphes"), s["body"])

    story += [
        Paragraph("<b>Tableau 7</b> — SMR par décile (~10 % exposition chacun)", s["cap"]),
        _mt(
            [["Tranche", "Expo", "Dobs", "Datt", "SMR", "IC inf", "IC sup"]]
            + [
                [d["tranche_label"], f"{d['exposure']:.0f}", str(d["deaths_observed"]),
                 f"{d['deaths_expected']:.0f}", f"{d['smr']:.3f}",
                 f"{d['smr_ic_inf']:.3f}", f"{d['smr_ic_sup']:.3f}"]
                for d in dec
            ],
            [AW * 0.14] * 7,
        ),
        Spacer(1, 4 * mm),
    ]

    for alerte in (sec5.get("alertes") or []):
        if alerte and alerte.strip():
            story.append(Paragraph(f"⚠ {_esc(alerte.strip())}", s["warn"]))

    story += [
        Paragraph("<b>Figure 4</b> — SMR par décile", s["cap"]),
        RLImage(gs, width=AW, height=AW * 0.50),
        Spacer(1, 4 * mm),
        Paragraph("<b>Figure 5</b> — Décès observés vs attendus", s["cap"]),
        RLImage(gdc, width=AW, height=AW * 0.50),
        Spacer(1, 3 * mm),
        PageBreak(),
    ]

    # ── Section 6 — Conclusion ─────────────────────────────────────────────────
    sec6 = _sec("section_6_conclusion")
    story.append(Paragraph("6. CONCLUSION ET RECOMMANDATIONS", s["h1"]))
    for key in ("synthese", "recommandations", "validation"):
        val_text = sec6.get(key, "")
        if val_text:
            story.append(Paragraph(_esc(val_text), s["body"]))
    story.append(PageBreak())

    # ── Annexe ────────────────────────────────────────────────────────────────
    story += [
        Paragraph("ANNEXE — Paramètres et traçabilité", s["h1"]),
        _mt(
            [["Paramètre", "Valeur", "Description"]]
            + [[str(k), str(v), ""] for k, v in params.items()]
            + [
                ["Méthode", methode, mcfg["titre"]],
                ["Référence", pf["table_reference"], ""],
                ["Regroupement", "Déciles exposition", "~10 % par tranche"],
            ],
            [AW * 0.3, AW * 0.2, AW * 0.5],
        ),
    ]
    if tr:
        ti = [[k, str(v)] for k, v in tr.items() if k != "execution_log"]
        if ti:
            story += [
                Spacer(1, 4 * mm),
                Paragraph("<b>Traçabilité</b>", s["h2"]),
                _mt([["Clé", "Valeur"]] + ti, [AW * 0.3, AW * 0.7]),
            ]
    story.append(Paragraph("Généré par l'Agent actuariel IA — Mortality.", s["note"]))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    ap = os.path.abspath(output_path)
    doc = _RT(ap, pagesize=A4, leftMargin=ML, rightMargin=MR,
              topMargin=MT, bottomMargin=MB)
    doc.build(story)
    return ap
