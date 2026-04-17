"""
Génère le diagramme d'architecture WriterAgent en PDF.
Usage : python docs/generate_architecture_pdf.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Couleurs ──────────────────────────────────────────────────────────────────
C_DET   = "#E8F4E8"   # vert clair — déterministe
C_LLM   = "#E3F2FD"   # bleu clair — LLM
C_ALERT = "#FFF3E0"   # orange clair — alerte
C_USER  = "#F3E5F5"   # violet clair — utilisateur
BORDER_DET  = "#4CAF50"
BORDER_LLM  = "#2196F3"
BORDER_ALERT= "#FF9800"
BORDER_USER = "#9C27B0"
TXT = "#1A1A2E"

OUTPUT = Path(__file__).parent / "architecture_writer_agent.pdf"


def box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.5, bold_first=True):
    """Dessine une boîte arrondie avec texte multiligne."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=3,
    )
    ax.add_patch(patch)
    lines = text.strip().split("\n")
    if bold_first and lines:
        ax.text(x, y + (len(lines) - 1) * 0.012,
                lines[0], ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TXT, zorder=4)
        body = "\n".join(lines[1:])
        if body:
            ax.text(x, y - 0.022,
                    body, ha="center", va="center",
                    fontsize=fontsize - 0.5, color=TXT, zorder=4,
                    linespacing=1.4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=TXT, zorder=4, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, label="", color="#555555"):
    """Dessine une flèche avec label optionnel."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.015, my, label, fontsize=7.5, color=color,
                ha="left", va="center", style="italic", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(14, 20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Titre ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.975, "Architecture WriterAgent — Génération de rapport PDF",
            ha="center", va="top", fontsize=14, fontweight="bold", color=TXT)
    ax.text(0.5, 0.958, "Agent actuariat — Marc Juillard — 2026",
            ha="center", va="top", fontsize=9, color="#666666")

    # ── Positions Y (de haut en bas) ──────────────────────────────────────────
    Y = {
        "user":    0.920,
        "01":      0.840,
        "02":      0.740,
        "master":  0.660,
        "03":      0.570,
        "04":      0.460,
        "05":      0.355,
        "06":      0.250,
        "ok":      0.140,
        "minor":   0.140,
        "major":   0.140,
    }
    CX = 0.50   # centre principal
    W_MAIN = 0.52
    H = 0.072

    # ── Boîtes principales ────────────────────────────────────────────────────

    # Utilisateur
    box(ax, CX, Y["user"], W_MAIN, 0.045,
        "Demande utilisateur\n\"génère le rapport\"",
        C_USER, BORDER_USER, fontsize=9)

    # 01
    box(ax, CX, Y["01"], W_MAIN, H,
        "① 01_load_plan.py  —  Déterministe\n"
        "Charge le YAML · Résout les {{ placeholders }}\n"
        "Produit un plan structuré avec 1 prompt par section\n"
        "Indique ready / missing par section",
        C_DET, BORDER_DET)

    # 02
    box(ax, CX, Y["02"], W_MAIN, H,
        "② 02_validation_plan.py  —  LLM\n"
        "Pour chaque section : données présentes ET suffisantes ?\n"
        "Outils disponibles ?\n"
        "Produit : liste OK / KO par section",
        C_LLM, BORDER_LLM)

    # Master (latéral)
    box(ax, 0.17, (Y["02"] + Y["master"]) / 2, 0.24, 0.075,
        "MasterAgent\n→ BuilderAgent\ncalcule les données\nretour en ①",
        C_ALERT, BORDER_ALERT, fontsize=8)

    # 03
    box(ax, CX, Y["03"], W_MAIN, H,
        "③ 03_completion_plan.py  —  LLM + RAG\n"
        "Pour chaque section : appelle search_exemplars\n"
        "Si exemples trouvés → enrichit le prompt\n"
        "Si corpus vide → continue sans exemples",
        C_LLM, BORDER_LLM)

    # 04
    box(ax, CX, Y["04"], W_MAIN, H + 0.010,
        "④ 04_redaction.py  —  Boucle Python + LLM\n"
        "Pour chaque section du plan enrichi :\n"
        "  · Appelle les tools tableaux / graphiques\n"
        "  · Appelle GPT-4o avec prompt + données + exemples RAG\n"
        "  · Stocke le résultat dans section_outputs",
        C_DET, BORDER_DET)

    # 05
    box(ax, CX, Y["05"], W_MAIN, H,
        "⑤ 05_assemble.py  —  Déterministe\n"
        "Mise en page ReportLab · En-têtes, numérotation, TOC\n"
        "Intégration tableaux + graphiques\n"
        "Export PDF",
        C_DET, BORDER_DET)

    # 06
    box(ax, CX, Y["06"], W_MAIN, H,
        "⑥ 06_validation.py  —  LLM\n"
        "Compare demande initiale · rapport · standard professionnel\n"
        "[MINEURE] Mineure : style, longueur, formulation\n"
        "[MAJEURE] Majeure : section manquante, chiffres incohérents",
        C_LLM, BORDER_LLM)

    # Sorties
    box(ax, 0.20, Y["ok"], 0.24, 0.060,
        "Rapport PDF livré ✓\nAucune alerte",
        C_DET, BORDER_DET, fontsize=8)

    box(ax, 0.50, Y["minor"], 0.26, 0.060,
        "[MINEURE] Retry ciblé (max 1)\nRelance ④ pour la section\navec feedback validateur",
        C_ALERT, BORDER_ALERT, fontsize=8)

    box(ax, 0.81, Y["major"], 0.27, 0.060,
        "[MAJEURE] Rapport livré + flag\nListe des points non couverts\nDécision utilisateur",
        "#FFEBEE", "#F44336", fontsize=8)

    # ── Flèches principales ───────────────────────────────────────────────────
    arrow(ax, CX, Y["user"] - 0.023, CX, Y["01"] + H / 2)
    arrow(ax, CX, Y["01"] - H / 2,  CX, Y["02"] + H / 2)
    arrow(ax, CX, Y["02"] - H / 2,  CX, Y["03"] + H / 2, "Toutes sections OK")
    arrow(ax, CX, Y["03"] - H / 2,  CX, Y["04"] + (H + 0.010) / 2)
    arrow(ax, CX, Y["04"] - (H + 0.010) / 2, CX, Y["05"] + H / 2)
    arrow(ax, CX, Y["05"] - H / 2,  CX, Y["06"] + H / 2)

    # 06 → sorties
    arrow(ax, 0.36, Y["06"] - H / 2, 0.20, Y["ok"] + 0.030, "Aucune alerte", BORDER_DET)
    arrow(ax, 0.50, Y["06"] - H / 2, 0.50, Y["minor"] + 0.030, "[MINEURE] Mineure", BORDER_ALERT)
    arrow(ax, 0.64, Y["06"] - H / 2, 0.81, Y["major"] + 0.030, "[MAJEURE] Majeure", "#F44336")

    # Retry minor → 05
    ax.annotate(
        "", xy=(0.76, Y["05"]),
        xytext=(0.63, Y["minor"] + 0.030),
        arrowprops=dict(arrowstyle="-|>", color=BORDER_ALERT,
                        lw=1.2, connectionstyle="arc3,rad=-0.3"),
        zorder=2,
    )
    ax.text(0.80, (Y["05"] + Y["minor"]) / 2, "ré-assemble",
            fontsize=7, color=BORDER_ALERT, style="italic")

    # 02 → Master (KO)
    arrow(ax, CX - W_MAIN / 2, Y["02"], 0.17 + 0.12, (Y["02"] + Y["master"]) / 2,
          "Sections KO", BORDER_ALERT)

    # Master → 01
    ax.annotate(
        "", xy=(CX - W_MAIN / 2, Y["01"]),
        xytext=(0.17 + 0.12, (Y["02"] + Y["master"]) / 2 - 0.038),
        arrowprops=dict(arrowstyle="-|>", color=BORDER_ALERT,
                        lw=1.2, connectionstyle="arc3,rad=0.2"),
        zorder=2,
    )
    ax.text(0.07, (Y["01"] + Y["master"]) / 2, "retour en ①",
            fontsize=7, color=BORDER_ALERT, style="italic")

    # ── Légende ───────────────────────────────────────────────────────────────
    legend_y = 0.058
    patches = [
        mpatches.Patch(facecolor=C_DET,   edgecolor=BORDER_DET,   label="Déterministe — Python pur, zéro token LLM"),
        mpatches.Patch(facecolor=C_LLM,   edgecolor=BORDER_LLM,   label="LLM — GPT-4o pour juger, enrichir ou rédiger"),
        mpatches.Patch(facecolor=C_ALERT, edgecolor=BORDER_ALERT, label="Routage conditionnel"),
    ]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, 0.005),
              ncol=3, fontsize=8, frameon=True, framealpha=0.9)

    # ── Export ────────────────────────────────────────────────────────────────
    fig.tight_layout(pad=0.5)
    fig.savefig(str(OUTPUT), format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"PDF généré : {OUTPUT}")


if __name__ == "__main__":
    main()
