"""Script 4 — análise do experimento principal (Seção 4.3 do TCC 2).

Agrega as 5 repetições de cada configuração em média e desvio-padrão (passo e)
da Seção 3.5). As tabelas daqui são as de trabalho; as versões exatamente
como aparecem no TCC (Tabelas 5, 6, 7 e 11, com as legendas do texto) são
geradas pelo 08_tabelas_figuras_tcc.py, que reaproveita as funções deste
script.

Lê <saida>/resultados_execucoes.csv e gera em <saida>/analise:
  - resumo_metricas.csv          (média, desvio-padrão e n por configuração)
  - tabela_metricas.tex          (precisão, recall, F1, PR-AUC, ROC-AUC, acurácia, TFP)
  - tabela_matriz_confusao.tex   (VP, FP, FN, VN médios)
  - tabela_custo.tex             (tempos de SMOTE-NC, treino e predição)
  - figuras/curvas_<config>.png/.pdf       ROC e PR das 5 sementes
  - figuras/comparacao_<cenario>.png/.pdf  ROC e PR médias das 4 configurações
  - figuras/importancia_<cenario>.png/.pdf importância das características (RF)

Uso:
  python 04_analise.py --saida resultados
  python 04_analise.py --saida resultados --decimal ponto   # 0.9876 em vez de 0,9876
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICAS = [("precisao", "Precisão"), ("recall", "Sensibilidade"), ("f1", "F1"),
            ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("acuracia", "Acurácia"),
            ("taxa_falsos_positivos", "TFP")]
CONFUSAO = [("vp", "VP"), ("fp", "FP"), ("fn", "FN"), ("vn", "VN")]
TEMPOS = [("tempo_smotenc_s", "SMOTE-NC (s)"), ("tempo_treino_s", "Treino (s)"),
          ("tempo_predicao_s", "Predição (s)")]
NOME_MODELO = {"DT": "Árvore de Decisão", "RF": r"\textit{Random Forest}"}
NOME_MODELO_TXT = {"DT": "Árvore de Decisão", "RF": "Random Forest"}
NOME_BAL = {"original": "Original", "smotenc": "SMOTE-NC"}
NOME_CONJ = {"stateless14": "", "stateless_sem_sld": " (sem sld)"}
ORDEM = [(c, m, b) for c in ["light", "heavy"] for m in ["DT", "RF"] for b in ["original", "smotenc"]]

# Paleta categórica (ordem fixa) e tinta de texto
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
GRADE = "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fmt(v, casas, dec):
    if pd.isna(v):
        return "--"
    s = f"{v:.{casas}f}"
    return s.replace(".", ",") if dec == "virgula" else s


def ms(m, s, casas, dec):
    return f"{fmt(m, casas, dec)} $\\pm$ {fmt(s, casas, dec)}"


def rotulo(cfg, latex=True):
    nm = NOME_MODELO if latex else NOME_MODELO_TXT
    return f"{nm[cfg['modelo']]} / {NOME_BAL[cfg['balanceamento']]}{NOME_CONJ[cfg['conjunto_caracteristicas']]}"


def resumir(d: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio-padrão amostral (ddof=1) das repetições de cada configuração."""
    chaves =["id_config", "cenario", "modelo", "balanceamento", "conjunto_caracteristicas"]
    cols = [m for m, _ in METRICAS] + [c for c, _ in CONFUSAO] + [t for t, _ in TEMPOS] + \
           ["tempo_predict_proba_s", "tempo_fit_total_s"]
    g = d.groupby(chaves, sort=False)[cols]
    r = g.mean().add_suffix("_media").join(g.std(ddof=1).add_suffix("_dp"))
    r["n_execucoes"] = g.size()
    r["sementes"] = d.groupby(chaves, sort=False).semente.apply(lambda s: " ".join(map(str, sorted(s))))
    ordem_cols = []
    for c in cols:
        ordem_cols += [c + "_media", c + "_dp"]
    r = r[ordem_cols + ["n_execucoes", "sementes"]].reset_index()
    pos = {k: i for i, k in enumerate(ORDEM)}
    r["_o"] = [pos.get((a, b, c), 99) * 10 + (k != "stateless14")
               for a, b, c, k in zip(r.cenario, r.modelo, r.balanceamento, r.conjunto_caracteristicas)]
    return r.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def tabela_latex(r, colunas, casas, titulo, rotulo_tab, dec, nota=None):
    L = [r"\begin{table}[htb]", r"  \centering", r"  \small",
         f"  \\caption{{{titulo}}}", f"  \\label{{{rotulo_tab}}}",
         r"  \resizebox{\textwidth}{!}{%",
         "  \\begin{tabular}{ll" + "c" * len(colunas) + "}", r"    \hline",
         "    Cenário & Configuração & " + " & ".join(n for _, n in colunas) + r" \\",
         r"    \hline"]
    ultimo = None
    for _, cfg in r.iterrows():
        cen = f"\\textit{{{cfg.cenario.capitalize()}}}" if cfg.cenario != ultimo else ""
        if ultimo is not None and cfg.cenario != ultimo:
            L.append(r"    \hline")
        ultimo = cfg.cenario
        vals = [ms(cfg[c + "_media"], cfg[c + "_dp"], casas, dec) for c, _ in colunas]
        L.append(f"    {cen} & {rotulo(cfg)} & " + " & ".join(vals) + r" \\")
    L += [r"    \hline", r"  \end{tabular}}"]
    leg = "Fonte: Autoria própria (2026)."
    if nota:
        leg = nota + " " + leg
    L += [f"  \\legend{{{leg}}}", r"\end{table}", ""]
    return "\n".join(L)


def ler_curvas(pasta, idc, tipo):
    out = []
    for f in sorted(glob.glob(os.path.join(pasta, "curvas", f"{idc}__s*_{tipo}.csv.gz"))):
        s = int(f.rsplit("__s", 1)[1].split("_")[0])
        out.append((s, pd.read_csv(f)))
    return out


def roc_media(curvas, grade):
    tprs = []
    for _, c in curvas:
        t = np.interp(grade, c.fpr.values, c.tpr.values)
        t[0] = 0.0
        tprs.append(t)
    return np.mean(tprs, axis=0)


def pr_media(curvas, grade):
    precs = []
    for _, c in curvas:
        rc = c.recall.values[::-1]
        pr = np.maximum.accumulate(c.precisao.values)[::-1]  # precisão interpolada
        precs.append(np.interp(grade, rc, pr))
    return np.mean(precs, axis=0)


def salvar(fig, caminho):
    fig.savefig(caminho + ".png")
    fig.savefig(caminho + ".pdf")
    plt.close(fig)


def fig_por_config(pasta, r, d, saida_fig):
    for _, cfg in r.iterrows():
        roc = ler_curvas(pasta, cfg.id_config, "roc")
        pr = ler_curvas(pasta, cfg.id_config, "pr")
        if not roc:
            continue
        prev = d[d.id_config == cfg.id_config].eval("n_teste_ataque / n_teste").mean()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
        for _, c in roc:
            a1.plot(c.fpr, c.tpr, color=CORES[0], lw=1.2, alpha=0.75)
        a1.plot([0, 1], [0, 1], ls="--", lw=1, color=TINTA2)
        a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
               xlim=(0, 1), ylim=(0, 1.01), title="Curva ROC")
        a1.text(0.98, 0.04, f"ROC-AUC = {cfg.roc_auc_media:.4f} ± {cfg.roc_auc_dp:.4f}\n"
                f"{len(roc)} sementes", ha="right", va="bottom", color=TINTA, fontsize=9)
        for _, c in pr:
            a2.plot(c.recall, c.precisao, color=CORES[0], lw=1.2, alpha=0.75,
                    drawstyle="steps-post")
        a2.axhline(prev, ls="--", lw=1, color=TINTA2)
        a2.text(0.02, prev + 0.015, f"proporção de ataques = {prev:.3f}", color=TINTA2, fontsize=8)
        a2.set(xlabel="Sensibilidade (recall)", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1.01),
               title="Curva Precision-Recall")
        a2.text(0.98, 0.04, f"PR-AUC = {cfg.pr_auc_media:.4f} ± {cfg.pr_auc_dp:.4f}",
                ha="right", va="bottom", color=TINTA, fontsize=9)
        fig.suptitle(f"{cfg.cenario.capitalize()} — {rotulo(cfg, latex=False)}", color=TINTA)
        fig.tight_layout()
        salvar(fig, os.path.join(saida_fig, f"curvas_{cfg.id_config}"))


def fig_comparacao(pasta, r, saida_fig):
    grade = np.linspace(0, 1, 501)
    for (cen, conj), sub in r.groupby(["cenario", "conjunto_caracteristicas"], sort=False):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
        desenhou = False
        for i, (_, cfg) in enumerate(sub.iterrows()):
            roc = ler_curvas(pasta, cfg.id_config, "roc")
            pr = ler_curvas(pasta, cfg.id_config, "pr")
            if not roc:
                continue
            desenhou = True
            cor = CORES[i % len(CORES)]
            ls = "-" if cfg.balanceamento == "original" else "--"
            nome = rotulo(cfg, latex=False).replace(NOME_CONJ[conj], "")
            a1.plot(grade, roc_media(roc, grade), color=cor, lw=2, ls=ls,
                    label=f"{nome} (AUC {cfg.roc_auc_media:.3f})")
            a2.plot(grade, pr_media(pr, grade), color=cor, lw=2, ls=ls,
                    label=f"{nome} (AP {cfg.pr_auc_media:.3f})")
        if not desenhou:
            plt.close(fig)
            continue
        a1.plot([0, 1], [0, 1], ls=":", lw=1, color=TINTA2)
        a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
               xlim=(0, 1), ylim=(0, 1.01), title="ROC (média das sementes)")
        a2.set(xlabel="Sensibilidade (recall)", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1.01),
               title="Precision-Recall (média das sementes)")
        a1.legend(fontsize=8, loc="lower right")
        a2.legend(fontsize=8, loc="lower left")
        fig.suptitle(f"Cenário {cen.capitalize()}{NOME_CONJ[conj]}", color=TINTA)
        fig.tight_layout()
        salvar(fig, os.path.join(saida_fig, f"comparacao_{cen}_{conj}"))


def fig_importancia(pasta, r, saida_fig, saida):
    arqs = glob.glob(os.path.join(pasta, "importancias", "*.csv"))
    if not arqs:
        return
    imp = pd.concat([pd.read_csv(f) for f in arqs], ignore_index=True)
    meta = r.set_index("id_config")[["cenario", "modelo", "balanceamento", "conjunto_caracteristicas"]]
    imp = imp.join(meta, on="id_config")
    imp = imp[imp.cenario.notna()]
    agg = (imp.groupby(["id_config", "cenario", "modelo", "balanceamento",
                        "conjunto_caracteristicas", "caracteristica"])
           .importancia.agg(["mean", "std", "count"]).reset_index())
    agg.to_csv(os.path.join(saida, "importancia_caracteristicas.csv"), index=False)
    rf = agg[agg.modelo == "RF"]
    for (cen, conj), sub in rf.groupby(["cenario", "conjunto_caracteristicas"], sort=False):
        bals = [b for b in ["original", "smotenc"] if b in set(sub.balanceamento)]
        ordem = (sub[sub.balanceamento == bals[0]].sort_values("mean").caracteristica.tolist())
        y = np.arange(len(ordem))
        h = 0.8 / len(bals)
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(ordem) + 1.2))
        for j, b in enumerate(bals):
            s = sub[sub.balanceamento == b].set_index("caracteristica").reindex(ordem)
            ax.barh(y + ((len(bals) - 1) / 2 - j) * h, s["mean"], height=h * 0.92,
                    xerr=s["std"].fillna(0), color=CORES[j], label=NOME_BAL[b],
                    error_kw={"ecolor": TINTA2, "lw": 0.8, "capsize": 2})
        ax.set_yticks(y, ordem)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Importância média (redução de impureza) ± desvio-padrão")
        ax.set_title(f"Random Forest — cenário {cen.capitalize()}{NOME_CONJ[conj]}", color=TINTA)
        ax.legend(loc="lower right")
        fig.tight_layout()
        salvar(fig, os.path.join(saida_fig, f"importancia_{cen}_{conj}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="resultados",
                    help="pasta onde está resultados_execucoes.csv (ex.: resultados ou resultados/piloto)")
    ap.add_argument("--decimal", choices=["virgula", "ponto"], default="virgula")
    a = ap.parse_args()
    pasta = a.saida
    csv_ = os.path.join(pasta, "resultados_execucoes.csv")
    if not os.path.exists(csv_):
        raise SystemExit(f"Não encontrei {csv_}")
    d = pd.read_csv(csv_)
    dup = d.duplicated(["id_config", "semente"], keep="last")
    if dup.any():
        print(f"Aviso: {dup.sum()} linhas repetidas (config, semente); mantida a última.")
        d = d[~dup]
    saida = os.path.join(pasta, "analise")
    saida_fig = os.path.join(saida, "figuras")
    os.makedirs(saida_fig, exist_ok=True)

    r = resumir(d)
    r.to_csv(os.path.join(saida, "resumo_metricas.csv"), index=False)
    incompletas = r[r.n_execucoes < 5]
    nota = "Média $\\pm$ desvio-padrão de 5 repetições."
    if len(incompletas):
        print("Aviso: configurações com menos de 5 execuções:")
        print(incompletas[["id_config", "n_execucoes", "sementes"]].to_string(index=False))
        nota = "Média $\\pm$ desvio-padrão das repetições disponíveis (ver resumo\\_metricas.csv)."

    principal = r[r.conjunto_caracteristicas == "stateless14"]
    extra = r[r.conjunto_caracteristicas != "stateless14"]
    tex = []
    for sub, suf, tit in [(principal, "", ""), (extra, "_sem_sld", " — Random Forest sem a coluna sld")]:
        if sub.empty:
            continue
        tex.append(("tabela_metricas" + suf, tabela_latex(
            sub, METRICAS, 4, "Métricas de detecção da classe de ataque no conjunto de teste" + tit,
            "tab:metricas" + suf, a.decimal, nota)))
        tex.append(("tabela_matriz_confusao" + suf, tabela_latex(
            sub, CONFUSAO, 1, "Matriz de confusão média no conjunto de teste" + tit,
            "tab:confusao" + suf, a.decimal, nota)))
        tex.append(("tabela_custo" + suf, tabela_latex(
            sub, TEMPOS, 3, "Custo computacional (segundos)" + tit,
            "tab:custo" + suf, a.decimal,
            nota + f" Tempos com n\\_jobs = {int(d.n_jobs.iloc[0])}.")))
    for nome, conteudo in tex:
        with open(os.path.join(saida, nome + ".tex"), "w", encoding="utf-8") as fh:
            fh.write("% Requer \\usepackage{graphicx} (\\resizebox)\n" + conteudo)

    fig_por_config(pasta, r, d, saida_fig)
    fig_comparacao(pasta, r, saida_fig)
    fig_importancia(pasta, r, saida_fig, saida)

    pd.set_option("display.width", 200)
    print(r[["id_config", "n_execucoes", "precisao_media", "recall_media", "f1_media", "f1_dp",
             "pr_auc_media", "roc_auc_media", "tempo_treino_s_media", "tempo_smotenc_s_media"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArquivos gerados em {os.path.abspath(saida)}")


if __name__ == "__main__":
    main()
