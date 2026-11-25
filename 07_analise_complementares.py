"""Script 7 — análise dos experimentos complementares (Seções 4.4 a 4.9 do TCC 2).

As tabelas daqui são as de trabalho, com mais colunas. As versões exatamente
como aparecem no TCC (Tabelas 8, 9 e 10, Figuras 3 a 6) são geradas pelo
08_tabelas_figuras_tcc.py, que reaproveita as funções deste script.

Lê <saida>/resultados_execucoes.csv (principal, só leitura) e
<saida>/complementares/<experimento>/ e gera em <saida>/complementares/analise:

  resumo_<experimento>.csv                   média, dp e n de cada configuração
  tabela_referencia.tex                      modelos x tabela de consulta (teto)
  tabela_divisao_grupo.tex                   divisão aleatória x por grupo
  tabela_divisao_sem_duplicatas.tex          (se rodado) aleatória x sem duplicatas
  tabela_importancia_<cenario>_<modelo>.tex  impureza x permutação
  importancia_permutacao_vs_impureza.csv     (inclui a permutação medida pelo F1)
  tabela_balanceamento.tex                   SMOTE-NC x class_weight x limiar
  tabela_balanceamento_custo.tex             tempos (inclui o ajuste do limiar)
  tabela_cruzado.tex                         treino em um cenário, teste no outro
  figuras/referencia_<cenario>.png/.pdf      ROC e PR médias: modelos x referência
  figuras/importancia_<cenario>_<modelo>.png/.pdf

Mesmo formato do 04_analise.py (média ± dp, vírgula decimal, \\resizebox,
"Fonte: Autoria própria (2026)").

Uso:
  python 07_analise_complementares.py --saida resultados
  python 07_analise_complementares.py --saida resultados_teste      # teste rápido
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AN = importlib.import_module("04_analise")  # funções e estilo das tabelas/figuras

CEN = {"light": r"\textit{Light}", "heavy": r"\textit{Heavy}"}
CEN_TXT = {"light": "Light", "heavy": "Heavy"}
NOME_VAR = {
    "original": "Original", "smotenc": "SMOTE-NC",
    "classweight": r"\textit{class\_weight}",
    "original_limiar": "Original", "smotenc_limiar": "SMOTE-NC",
    "classweight_limiar": r"\textit{class\_weight}",
}
NOME_VAR_TXT = {k: v.replace(r"\textit{class\_weight}", "class_weight")
                for k, v in NOME_VAR.items()}
NOME_REF = {"limiar05": "Tabela de consulta (limiar 0,5)",
            "limiarf1": "Tabela de consulta (limiar do treino)"}
FONTE = "Fonte: Autoria própria (2026)."
MET5 = [("precisao", "Precisão"), ("recall", "Sensibilidade"), ("f1", "F1"),
        ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]


# ---------------------------------------------------------------------------
def tabela(linhas, cab_rotulos, cab_valores, titulo, rotulo_tab, nota=None):
    """Mesmo layout de 04_analise.tabela_latex, com rótulos livres.

    linhas: lista de (rótulos, valores); o 1º rótulo só é escrito quando muda,
    e uma linha \\hline separa os grupos.
    """
    n_rot = len(cab_rotulos)
    L = [r"\begin{table}[htb]", r"  \centering", r"  \small",
         f"  \\caption{{{titulo}}}", f"  \\label{{{rotulo_tab}}}",
         r"  \resizebox{\textwidth}{!}{%",
         "  \\begin{tabular}{" + "l" * n_rot + "c" * len(cab_valores) + "}", r"    \hline",
         "    " + " & ".join(list(cab_rotulos) + list(cab_valores)) + r" \\", r"    \hline"]
    ultimo = None
    for rots, vals in linhas:
        rots = list(rots)
        if n_rot > 1:
            if ultimo is not None and rots[0] != ultimo:
                L.append(r"    \hline")
            grupo = rots[0]
            rots[0] = rots[0] if rots[0] != ultimo else ""
            ultimo = grupo
        L.append("    " + " & ".join(rots + list(vals)) + r" \\")
    L += [r"    \hline", r"  \end{tabular}}"]
    leg = FONTE if not nota else nota + " " + FONTE
    L += [f"  \\legend{{{leg}}}", r"\end{table}", ""]
    return "% Requer \\usepackage{graphicx} (\\resizebox)\n" + "\n".join(L)


def ler(caminho):
    if not os.path.exists(caminho):
        return None
    d = pd.read_csv(caminho)
    return d[~d.duplicated(["id_config", "semente"], keep="last")]


def resumo(d):
    """04_analise.resumir + colunas extras (limiar, tempo do ajuste, prevalência)."""
    r = AN.resumir(d).set_index("id_config")
    d = d.assign(prop_ataque_teste=d.n_teste_ataque / d.n_teste)
    extras = [c for c in ["limiar", "tempo_ajuste_limiar_s", "prop_ataque_teste",
                          "n_teste_ataque", "n_linhas_teste_vetor_inedito"] if c in d]
    g = d.groupby("id_config")[extras]
    r = r.join(g.mean().add_suffix("_media")).join(g.std(ddof=1).add_suffix("_dp"))
    return r


def val(r, idc, col, casas, dec):
    if r is None or idc not in r.index:
        return "--"
    return AN.ms(r.loc[idc, col + "_media"], r.loc[idc, col + "_dp"], casas, dec)


def idc(cen, mod, var, conj="stateless14"):
    return f"{cen}__{mod}__{var}__{conj}"


def rot_config(mod, var):
    if mod == "REF":
        return NOME_REF[var]
    return f"{AN.NOME_MODELO[mod]} / {NOME_VAR[var]}"


def nota_n(*resumos):
    ns = sorted({int(n) for r in resumos if r is not None for n in r.n_execucoes})
    if ns == [5]:
        return "Média $\\pm$ desvio-padrão de 5 repetições."
    return ("Média $\\pm$ desvio-padrão das repetições disponíveis "
            f"({', '.join(map(str, ns))}; ver os arquivos resumo\\_*.csv).")


def escrever(saida, nome, conteudo, gerados):
    with open(os.path.join(saida, nome), "w", encoding="utf-8") as fh:
        fh.write(conteudo)
    gerados.append(nome)


# ---------------------------------------------------------------------------
# 1. Referência — Seção 4.4 (Tabela 8, Figuras 3 e 4)
# ---------------------------------------------------------------------------
def analise_referencia(base, rp, rr, dec, saida, gerados):
    linhas = []
    for cen in ["light", "heavy"]:
        for m in ["DT", "RF"]:
            for b in ["original", "smotenc"]:
                i = idc(cen, m, b)
                linhas.append(([CEN[cen], rot_config(m, b)],
                               [val(rp, i, c, 4, dec) for c, _ in MET5]))
        for v in ["limiar05", "limiarf1"]:
            i = idc(cen, "REF", v)
            linhas.append(([CEN[cen], rot_config("REF", v)],
                           [val(rr, i, c, 4, dec) for c, _ in MET5]))
    nota = (nota_n(rp, rr) + " A tabela de consulta atribui a cada vetor das 14 "
            "características a proporção de ataque desse vetor no treino (vetores ausentes "
            "do treino recebem a proporção geral). Limiar 0,5: classe majoritária do vetor; "
            "limiar do treino: o de maior F1 no conjunto de treino. Mesma divisão 70/30 e "
            "mesmas sementes dos modelos.")
    escrever(saida, "tabela_referencia.tex", tabela(
        linhas, ["Cenário", "Configuração"], [n for _, n in MET5],
        "Modelos e linha de referência (tabela de consulta) no conjunto de teste",
        "tab:referencia", nota), gerados)

    # figura: ROC e PR médias, 4 modelos + referência
    pasta_ref = os.path.join(base, "complementares", "referencia")
    grade = np.linspace(0, 1, 501)
    for cen in ["light", "heavy"]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
        k = 0
        for m in ["DT", "RF"]:
            for b in ["original", "smotenc"]:
                roc = AN.ler_curvas(base, idc(cen, m, b), "roc")
                pr = AN.ler_curvas(base, idc(cen, m, b), "pr")
                if not roc:
                    continue
                nome = f"{AN.NOME_MODELO_TXT[m]} / {NOME_VAR_TXT[b]}"
                ls = "-" if b == "original" else "--"
                a1.plot(grade, AN.roc_media(roc, grade), color=AN.CORES[k], lw=1.6, ls=ls,
                        label=nome)
                a2.plot(grade, AN.pr_media(pr, grade), color=AN.CORES[k], lw=1.6, ls=ls,
                        label=nome)
                k += 1
        roc = AN.ler_curvas(pasta_ref, idc(cen, "REF", "limiarf1"), "roc")
        pr = AN.ler_curvas(pasta_ref, idc(cen, "REF", "limiarf1"), "pr")
        if roc:
            a1.plot(grade, AN.roc_media(roc, grade), color=AN.TINTA, lw=1.2, ls=":",
                    label="Tabela de consulta")
            a2.plot(grade, AN.pr_media(pr, grade), color=AN.TINTA, lw=1.2, ls=":",
                    label="Tabela de consulta")
        a1.plot([0, 1], [0, 1], ls=":", lw=0.8, color=AN.TINTA2)
        a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
               xlim=(0, 1), ylim=(0, 1.01), title="ROC (média das sementes)")
        a2.set(xlabel="Sensibilidade (recall)", ylabel="Precisão", xlim=(0, 1),
               ylim=(0, 1.01), title="Precision-Recall (média das sementes)")
        a1.legend(fontsize=8, loc="lower right")
        a2.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Cenário {CEN_TXT[cen]} — modelos e tabela de consulta", color=AN.TINTA)
        fig.tight_layout()
        AN.salvar(fig, os.path.join(saida, "figuras", f"referencia_{cen}"))
        gerados.append(f"figuras/referencia_{cen}")


# ---------------------------------------------------------------------------
# 2. Divisão por grupo / sem duplicatas — Seção 4.5 (Tabela 9)
# ---------------------------------------------------------------------------
def analise_divisao(exp, rp, rr, rd, dec, saida, gerados):
    nome_d = {"grupo": "grupo", "sem_duplicatas": "s/ dupl."}[exp]
    cols = []
    for c, n in [("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]:
        cols += [(c, "aleat", f"{n} (aleat.)"), (c, exp, f"{n} ({nome_d})")]
    linhas = []
    for cen in ["light", "heavy"]:
        for m in ["DT", "RF"]:
            for b in ["original", "smotenc"]:
                i = idc(cen, m, b)
                linhas.append(([CEN[cen], rot_config(m, b)],
                               [val(rp if q == "aleat" else rd, i, c, 4, dec)
                                for c, q, _ in cols]))
        i = idc(cen, "REF", "limiarf1")
        linhas.append(([CEN[cen], rot_config("REF", "limiarf1")],
                       [val(rr if q == "aleat" else rd, i, c, 4, dec) for c, q, _ in cols]))
    if exp == "grupo":
        tit = "Divisão aleatória (70/30) e validação cruzada por grupo"
        expl = ("Por grupo: StratifiedGroupKFold com 3 dobras, sendo o grupo o vetor das 14 "
                "características (nenhum vetor do teste aparece no treino); métricas sobre a "
                "predição fora da dobra de toda a base experimental, uma por semente. "
                "Hiperparâmetros do ajuste principal.")
    else:
        tit = "Divisão aleatória com e sem linhas duplicadas"
        prev = rd.prop_ataque_teste_media if "prop_ataque_teste_media" in rd else None
        expl = ("Sem duplicatas: linhas repetidas (14 características + classe) removidas "
                "antes da divisão aleatória 70/30.")
        if prev is not None:
            expl += (" Proporção de ataque no teste: "
                     f"{AN.fmt(prev.min(), 4, dec)} a {AN.fmt(prev.max(), 4, dec)}.")
    escrever(saida, f"tabela_divisao_{exp}.tex", tabela(
        linhas, ["Cenário", "Configuração"], [n for *_, n in cols], tit,
        f"tab:divisao_{exp}", nota_n(rp, rd) + " " + expl), gerados)


# ---------------------------------------------------------------------------
# 3. Importância: impureza x permutação — Seção 4.7 (Figuras 5 e 6)
# ---------------------------------------------------------------------------
def analise_importancia(base, dec, saida, gerados):
    arqs_p = glob.glob(os.path.join(base, "complementares", "permutacao",
                                    "importancias_permutacao", "*.csv"))
    arqs_m = glob.glob(os.path.join(base, "importancias", "*.csv"))
    if not arqs_p:
        return
    perm = pd.concat([pd.read_csv(f) for f in arqs_p], ignore_index=True)
    perm = (perm.groupby(["id_config", "metrica", "caracteristica"]).importancia_media
            .agg(["mean", "std", "count"]).reset_index())
    mdi = pd.concat([pd.read_csv(f) for f in arqs_m], ignore_index=True)
    mdi = (mdi.groupby(["id_config", "caracteristica"]).importancia
           .agg(["mean", "std", "count"]).reset_index().assign(metrica="impureza"))
    tudo = pd.concat([mdi, perm], ignore_index=True)
    tudo = tudo[tudo.id_config.isin(perm.id_config.unique())]
    tudo.to_csv(os.path.join(saida, "importancia_permutacao_vs_impureza.csv"), index=False)
    gerados.append("importancia_permutacao_vs_impureza.csv")

    def serie(i, metrica):
        s = tudo[(tudo.id_config == i) & (tudo.metrica == metrica)]
        return s.set_index("caracteristica")

    for cen in ["light", "heavy"]:
        for m in ["DT", "RF"]:
            ids = {b: idc(cen, m, b) for b in ["original", "smotenc"]}
            if not all(i in set(perm.id_config) for i in ids.values()):
                continue
            ordem = (serie(ids["original"], "pr_auc").sort_values("mean", ascending=False)
                     .index.tolist())
            linhas, rhos = [], []
            for car in ordem:
                vals = []
                for b in ["original", "smotenc"]:
                    for met in ["impureza", "pr_auc"]:
                        s = serie(ids[b], met)
                        vals.append(AN.ms(s.loc[car, "mean"], s.loc[car, "std"], 4, dec)
                                    if car in s.index else "--")
                linhas.append(([car.replace("_", r"\_")], vals))
            for b in ["original", "smotenc"]:
                a, p = serie(ids[b], "impureza"), serie(ids[b], "pr_auc")
                comum = a.index.intersection(p.index)
                rhos.append(spearmanr(a.loc[comum, "mean"], p.loc[comum, "mean"]).statistic)
            nota = ("Média $\\pm$ desvio-padrão entre as sementes."
                    " Impureza: redução média de impureza (Gini). Permutação: queda da PR-AUC "
                    "ao embaralhar a característica (10 repetições por semente); valores "
                    "negativos ou próximos de zero indicam que a característica não é usada. "
                    "Correlação de Spearman entre os dois ordenamentos: "
                    f"{AN.fmt(rhos[0], 2, dec)} (Original) e {AN.fmt(rhos[1], 2, dec)} "
                    "(SMOTE-NC).")
            escrever(saida, f"tabela_importancia_{cen}_{m}.tex", tabela(
                linhas, ["Característica"],
                ["Impureza (Original)", "Permutação (Original)", "Impureza (SMOTE-NC)",
                 "Permutação (SMOTE-NC)"],
                f"Importância por impureza e por permutação — {AN.NOME_MODELO[m]}, "
                f"cenário {CEN[cen]}", f"tab:importancia_{cen}_{m}", nota), gerados)

            # figura: dois painéis (impureza | permutação), barras Original e SMOTE-NC
            y = np.arange(len(ordem))[::-1]
            fig, axs = plt.subplots(1, 2, figsize=(10, 0.32 * len(ordem) + 1.4), sharey=True)
            for ax, met, tit in [(axs[0], "impureza", "Redução de impureza"),
                                 (axs[1], "pr_auc", "Permutação (queda da PR-AUC)")]:
                for j, b in enumerate(["original", "smotenc"]):
                    s = serie(ids[b], met).reindex(ordem)
                    ax.barh(y + (0.2 - 0.4 * j), s["mean"], height=0.37, xerr=s["std"].fillna(0),
                            color=AN.CORES[j], label=AN.NOME_BAL[b],
                            error_kw={"ecolor": AN.TINTA2, "lw": 0.8, "capsize": 2})
                ax.axvline(0, color=AN.TINTA2, lw=0.8)
                ax.set_title(tit, color=AN.TINTA)
                ax.grid(axis="y", visible=False)
                ax.xaxis.set_major_locator(MaxNLocator(5))
            axs[0].set_yticks(y, ordem)
            fig.suptitle(f"{AN.NOME_MODELO_TXT[m]} — cenário {CEN_TXT[cen]}", color=AN.TINTA)
            h, l = axs[0].get_legend_handles_labels()
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.legend(h, l, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99))
            AN.salvar(fig, os.path.join(saida, "figuras", f"importancia_{cen}_{m}"))
            gerados.append(f"figuras/importancia_{cen}_{m}")


# ---------------------------------------------------------------------------
# 4. Balanceamento x limiar x class_weight — Seção 4.6 (Tabela 10)
# ---------------------------------------------------------------------------
def analise_balanceamento(rp, rb, dec, saida, gerados):
    combos = [("original", False), ("smotenc", False), ("classweight", False),
              ("original_limiar", True), ("smotenc_limiar", True),
              ("classweight_limiar", True)]
    cols = MET5[:4] + [("taxa_falsos_positivos", "TFP"), ("limiar", "Limiar")]
    custo = [("tempo_smotenc_s", "SMOTE-NC (s)"), ("tempo_treino_s", "Treino (s)"),
             ("tempo_ajuste_limiar_s", "Ajuste do limiar (s)"),
             ("tempo_predicao_s", "Predição (s)")]
    linhas, linhas_c = [], []
    for cen in ["light", "heavy"]:
        for m in ["DT", "RF"]:
            for var, ajust in combos:
                i = idc(cen, m, var)
                r = rp if var in ("original", "smotenc") else rb
                rot = [CEN[cen], AN.NOME_MODELO[m], NOME_VAR[var]]
                vs = []
                for c, _ in cols:
                    if c == "limiar" and not ajust:
                        vs.append(AN.fmt(0.5, 1, dec))
                    else:
                        vs.append(val(r, i, c, 4, dec))
                linhas.append((rot, vs))
                vc = []
                for c, _ in custo:
                    if c == "tempo_ajuste_limiar_s" and not ajust:
                        vc.append("--")
                    else:
                        vc.append(val(r, i, c, 3, dec))
                linhas_c.append((rot, vc))
    nota = (nota_n(rp, rb) + " Linhas Original e SMOTE-NC com limiar 0,5: experimento "
            "principal. \\textit{class\\_weight} = `balanced' (pesos inversamente "
            "proporcionais às classes), com os hiperparâmetros da configuração Original. "
            "Limiar ajustado: o de maior F1 na predição fora da dobra de uma validação "
            "cruzada estratificada de 5 dobras no treino (o teste não é usado); limiar 0,5 = "
            "predição padrão do classificador.")
    escrever(saida, "tabela_balanceamento.tex", tabela(
        linhas, ["Cenário", "Modelo", "Balanceamento"], [n for _, n in cols],
        "Balanceamento por SMOTE-NC, por pesos de classe e por ajuste do limiar",
        "tab:balanceamento", nota), gerados)
    escrever(saida, "tabela_balanceamento_custo.tex", tabela(
        linhas_c, ["Cenário", "Modelo", "Balanceamento"], [n for _, n in custo],
        "Custo computacional do balanceamento e do ajuste do limiar (segundos)",
        "tab:balanceamento_custo", nota_n(rp, rb) + " Ajuste do limiar: 5 ajustes "
        "adicionais do pipeline (validação cruzada no treino)."), gerados)


# ---------------------------------------------------------------------------
# 5. Cruzado — Seção 4.9 (números citados no texto)
# ---------------------------------------------------------------------------
def analise_cruzado(rp, rr, rc, dec, saida, gerados):
    cols = [("prop_ataque_teste", "Ataque no teste")] + MET5
    linhas = []
    for o, d in [("heavy", "light"), ("light", "heavy")]:
        par = f"{o}_para_{d}"
        grupo = f"{CEN[o]} $\\rightarrow$ {CEN[d]}"
        confs = [(m, b) for m in ["DT", "RF"] for b in ["original", "smotenc"]] + \
                [("REF", "limiar05"), ("REF", "limiarf1")]
        for m, b in confs:
            i = idc(par, m, b)
            vs = [val(rc, i, c, 4, dec) for c, _ in cols]
            vs.append(val(rr if m == "REF" else rp, idc(o, m, b), "f1", 4, dec))
            linhas.append(([grupo, rot_config(m, b)], vs))
    nota = (nota_n(rc) + " Modelos treinados como no experimento principal (mesma divisão e "
            "mesmas sementes) e testados na base experimental do outro cenário, sem os "
            "arquivos benign\\_1 e benign\\_2, que pertencem aos dois cenários. F1 na origem: "
            "teste do próprio cenário de treino.")
    escrever(saida, "tabela_cruzado.tex", tabela(
        linhas, ["Treino $\\rightarrow$ teste", "Configuração"],
        [n for _, n in cols] + ["F1 na origem"],
        "Generalização entre cenários", "tab:cruzado", nota), gerados)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--decimal", choices=["virgula", "ponto"], default="virgula")
    a = ap.parse_args()
    base, dec = a.saida, a.decimal
    pc = os.path.join(base, "complementares")
    saida = os.path.join(pc, "analise")
    os.makedirs(os.path.join(saida, "figuras"), exist_ok=True)

    dp = ler(os.path.join(base, "resultados_execucoes.csv"))
    if dp is None:
        raise SystemExit(f"Não encontrei {base}/resultados_execucoes.csv")
    rp = resumo(dp)
    rs = {}
    for exp in ["referencia", "grupo", "sem_duplicatas", "balanceamento", "cruzado"]:
        d = ler(os.path.join(pc, exp, "resultados_execucoes.csv"))
        if d is not None:
            rs[exp] = resumo(d)
            rs[exp].reset_index().to_csv(os.path.join(saida, f"resumo_{exp}.csv"),
                                         index=False)
    gr = os.path.join(pc, "grupo", "resultados_por_dobra.csv")
    if os.path.exists(gr):
        g = pd.read_csv(gr)
        g = g[~g.duplicated(["id_config", "semente", "dobra"], keep="last")]
        (g.groupby(["id_config", "dobra"])[["n_teste", "n_teste_ataque", "f1", "pr_auc",
                                             "roc_auc"]]
         .agg(["mean", "std"]).to_csv(os.path.join(saida, "resumo_grupo_por_dobra.csv")))

    gerados = []
    rr = rs.get("referencia")
    if rr is not None:
        analise_referencia(base, rp, rr, dec, saida, gerados)
    for exp in ["grupo", "sem_duplicatas"]:
        if exp in rs:
            analise_divisao(exp, rp, rr, rs[exp], dec, saida, gerados)
    analise_importancia(base, dec, saida, gerados)
    if "balanceamento" in rs:
        analise_balanceamento(rp, rs["balanceamento"], dec, saida, gerados)
    if "cruzado" in rs:
        analise_cruzado(rp, rr, rs["cruzado"], dec, saida, gerados)

    pd.set_option("display.width", 200)
    for exp, r in rs.items():
        print(f"\n== {exp} ==")
        print(r[["n_execucoes", "precisao_media", "recall_media", "f1_media", "f1_dp",
                 "pr_auc_media", "roc_auc_media"]]
              .to_string(float_format=lambda v: f"{v:.4f}"))
    conf = os.path.join(pc, "conferencias_com_divergencia.txt")
    if os.path.exists(conf):
        print(f"\nATENÇÃO: há execuções refeitas que não reproduzem o principal ({conf}).")
    print(f"\nGerados em {os.path.abspath(saida)}:\n  " + "\n  ".join(gerados))


if __name__ == "__main__":
    main()
