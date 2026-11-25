"""Script 8 — tabelas e figuras do Capítulo 4 (Resultados) do TCC 2, prontas para o LaTeX.

Não roda nenhum modelo: só lê o que os scripts 01, 03 e 06 gravaram em
<saida>/ e monta cada tabela e figura com o título, o rótulo (\\label) e a
legenda usados no texto. A numeração é a da versão do TCC 2 de novembro de 2026.

  Tabela 3   Hiperparâmetros selecionados pelo Grid Search      Seção 4.1   hiperparametros/*.json (03)
  Tabela 4   Vetores de características por classe              Seção 4.2   verificacao/resumo_vetores.csv (01)
  Tabela 5   Métricas de detecção no conjunto de teste          Seção 4.3   resultados_execucoes.csv (03)
  Tabela 6   Matriz de confusão média                           Seção 4.3   resultados_execucoes.csv (03)
  Tabela 7   Custo computacional                                Seção 4.3   resultados_execucoes.csv (03)
  Tabela 8   Linha de referência (tabela de consulta)           Seção 4.4   complementares/referencia (06)
  Tabela 9   Divisão aleatória x validação por grupos           Seção 4.5   complementares/grupo (06)
  Tabela 10  F1 por balanceamento e limiar de decisão           Seção 4.6   complementares/balanceamento (06)
  Tabela 11  Random Forest com e sem a característica sld       Seção 4.8   resultados_execucoes.csv (03 --sem-sld)
  Figura 3   Curvas ROC e Precision-Recall, cenário light       Seção 4.4   curvas/ e complementares/referencia
  Figura 4   Curvas ROC e Precision-Recall, cenário heavy       Seção 4.4   idem
  Figura 5   Importância das características, RF, light         Seção 4.7   importancias/ e complementares/permutacao
  Figura 6   Importância das características, DT, light         Seção 4.7   idem

As Tabelas 1 e 2 (Cap. 3) não vêm dos experimentos: a 1 é adaptada do artigo
do conjunto de dados (o script 01 confere as contagens) e a 2 lista as
configurações. Os números da Seção 4.9 aparecem só no texto; estão em
complementares/analise/tabela_cruzado.tex (script 07).

Saída em <saida>/tcc/:
  tabela_03_hiperparametros.tex ... tabela_11_sem_sld.tex
  figuras/referencia_light.pdf/.png, referencia_heavy, importancia_light_RF,
  importancia_light_DT (mesmos nomes usados nos \\includegraphics do capítulo)
  capitulo4_tabelas.tex (todas as tabelas em sequência, para conferência)

Uso:
  python 08_tabelas_figuras_tcc.py --saida resultados
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

AN = importlib.import_module("04_analise")              # resumir, fmt, ms, estilo
AC = importlib.import_module("07_analise_complementares")  # resumo, figuras

DEC = "virgula"                       # o TCC usa vírgula decimal
FONTE = "Fonte: Autoria própria (2026)."
CENARIOS = ["light", "heavy"]
CONFIGS = [(m, b) for m in ["DT", "RF"] for b in ["original", "smotenc"]]
CEN = {"light": r"\textit{Light}", "heavy": r"\textit{Heavy}"}


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------
def f(v, casas):
    """Número com vírgula decimal (ex.: 0,4268)."""
    return AN.fmt(v, casas, DEC)


def mdp(r, i, col, casas):
    """'média $\\pm$ desvio-padrão' da coluna `col` da configuração `i`."""
    return AN.ms(r.loc[i, col + "_media"], r.loc[i, col + "_dp"], casas, DEC)


def milhar(n):
    """Inteiro com ponto de milhar (ex.: 281.164)."""
    return f"{int(n):,}".replace(",", ".")


def idc(cen, mod, var, conj="stateless14"):
    return f"{cen}__{mod}__{var}__{conj}"


def rotulo(m, b):
    return f"{AN.NOME_MODELO[m]} / {AN.NOME_BAL[b]}"


def tabela(titulo, label, colunas, cabecalho, grupos, legenda, agrupar=True):
    """Tabela no formato do capítulo (\\small, \\resizebox e \\legend).

    cabecalho: lista de linhas do cabeçalho (sem o '\\\\' final).
    grupos: lista de grupos; cada grupo é uma lista de linhas (listas de
    células). Com agrupar=True, o primeiro rótulo de cada grupo só aparece na
    primeira linha. Uma linha \\hline separa os grupos.
    """
    L = [r"\begin{table}[htb]", r"  \centering", r"  \small",
         f"  \\caption{{{titulo}}}", f"  \\label{{{label}}}",
         r"  \resizebox{\textwidth}{!}{%", f"  \\begin{{tabular}}{{{colunas}}}",
         r"    \hline"]
    L += [f"    {c} \\\\" for c in cabecalho]
    for g in grupos:
        L.append(r"    \hline")
        for k, linha in enumerate(g):
            celulas = list(linha)
            if agrupar and k > 0:
                celulas[0] = ""
            L.append("    " + " & ".join(celulas) + r" \\")
    leg = f"{legenda} {FONTE}" if legenda else FONTE
    L += [r"    \hline", r"  \end{tabular}}", f"  \\legend{{{leg}}}",
          r"\end{table}", ""]
    return "\n".join(L)


def nota_repeticoes(*resumos):
    """Confere que cada configuração tem as 5 repetições previstas (Seção 3.5)."""
    ns = sorted({int(n) for r in resumos for n in r.n_execucoes})
    if ns != [5]:
        print(f"AVISO: há configurações com {ns} repetições (esperado: 5). "
              "As legendas dizem '5 repetições'; confira antes de usar.")


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def tabela_03(base):
    """Hiperparâmetros escolhidos no Grid Search (Seção 3.4) — Seção 4.1."""
    grupos = []
    for cen in CENARIOS:
        g = []
        for m, b in CONFIGS:
            caminho = os.path.join(base, "hiperparametros", f"{idc(cen, m, b)}.json")
            with open(caminho, encoding="utf-8") as fh:
                hp = json.load(fh)
            mh = hp["melhores_hiperparametros"]

            def v(k):
                if k not in mh:
                    return "--"
                return r"\texttt{None}" if mh[k] is None else str(mh[k])
            g.append([CEN[cen], rotulo(m, b), v("max_depth"), v("min_samples_split"),
                      v("n_estimators"),
                      AN.ms(hp["f1_cv_media"], hp["f1_cv_dp"], 4, DEC),
                      str(hp["combinacoes_empatadas_no_topo"])])
        grupos.append(g)
    return tabela(
        r"Hiperparâmetros selecionados pelo \textit{Grid Search}", "tab:hiperparametros",
        "llccccc",
        [r"Cenário & Configuração & \texttt{max\_depth} & \texttt{min\_samples\_split} & "
         r"\texttt{n\_estimators} & F1 (validação) & Empates"],
        grupos,
        r"F1 (validação): média $\pm$ desvio-padrão do \textit{F1-score} da classe de ataque "
        r"nas 5 dobras da validação cruzada. Empates: número de combinações da grade com o "
        r"mesmo F1 máximo; nesses casos, o \textit{Grid Search} adota a primeira delas na "
        r"ordem da grade.")


def tabela_04(base):
    """Vetores das 14 características por classe (script 01) — Seção 4.2."""
    caminho = os.path.join(base, "verificacao", "resumo_vetores.csv")
    if not os.path.exists(caminho):
        raise SystemExit(f"{caminho} não existe: rode antes o 01_verificar_dados.py.")
    v = pd.read_csv(caminho).set_index("cenario")

    def pc(x):
        return f(x, 1) + r"\%"
    linhas = [
        ("Amostras benignas / de ataque",
         lambda c: f"{milhar(v.loc[c, 'n_benigno'])} / {milhar(v.loc[c, 'n_ataque'])}"),
        ("Amostras com vetor repetido (duplicatas das 14 características)",
         lambda c: pc(v.loc[c, "pct_duplicatas_14"])),
        ("Vetores distintos na classe benigna", lambda c: milhar(v.loc[c, "vetores_benigno"])),
        ("Vetores distintos na classe de ataque", lambda c: milhar(v.loc[c, "vetores_ataque"])),
        ("Vetores presentes nas duas classes",
         lambda c: milhar(v.loc[c, "vetores_nas_duas_classes"])),
        ("Amostras de ataque cujo vetor também aparece como benigno",
         lambda c: f"{milhar(v.loc[c, 'ataque_com_vetor_benigno'])} "
                   f"({pc(v.loc[c, 'pct_ataque_com_vetor_benigno'])})"),
        ("Amostras benignas cujo vetor também aparece como ataque",
         lambda c: f"{milhar(v.loc[c, 'benigno_com_vetor_ataque'])} "
                   f"({pc(v.loc[c, 'pct_benigno_com_vetor_ataque'])})"),
    ]
    corpo = [[nome] + [fn(c) for c in CENARIOS] for nome, fn in linhas]
    return tabela("Vetores de características por classe", "tab:verificacao", "lcc",
                  [r" & \textit{Light} & \textit{Heavy}"], [corpo], "", agrupar=False)


def grupos_principal(rp, colunas, casas, traco_original=()):
    """Linhas das 8 configurações (Tabelas 5, 6 e 7)."""
    grupos = []
    for cen in CENARIOS:
        g = []
        for m, b in CONFIGS:
            i = idc(cen, m, b)
            vals = ["--" if (c in traco_original and b == "original") else mdp(rp, i, c, casas)
                    for c, _ in colunas]
            g.append([CEN[cen], rotulo(m, b)] + vals)
        grupos.append(g)
    return grupos


def tabela_05(rp):
    """Métricas da Seção 3.6 no conjunto de teste — Seção 4.3.1."""
    return tabela(
        "Métricas de detecção da classe de ataque no conjunto de teste", "tab:metricas",
        "llccccccc",
        ["Cenário & Configuração & " + " & ".join(n for _, n in AN.METRICAS)],
        grupos_principal(rp, AN.METRICAS, 4),
        r"Média $\pm$ desvio-padrão de 5 repetições. TFP: taxa de falsos positivos.")


def tabela_06(rp):
    """Matriz de confusão média — Seção 4.3.1."""
    return tabela(
        "Matriz de confusão média no conjunto de teste", "tab:confusao", "llcccc",
        ["Cenário & Configuração & VP & FP & FN & VN"],
        grupos_principal(rp, AN.CONFUSAO, 1),
        r"Média $\pm$ desvio-padrão de 5 repetições. VP: verdadeiros positivos; FP: falsos "
        r"positivos; FN: falsos negativos; VN: verdadeiros negativos.")


def tabela_07(rp, dp):
    """Tempos de SMOTE-NC, treino e predição (Seção 3.5) — Seção 4.3.2."""
    n_jobs = sorted(dp.n_jobs.unique())
    if len(n_jobs) != 1:
        print(f"AVISO: execuções com n_jobs diferentes ({n_jobs}); tempos não comparáveis.")
    n_teste = {c: dp[(dp.cenario == c)
                     & (dp.conjunto_caracteristicas == "stateless14")].n_teste.unique()
               for c in CENARIOS}
    return tabela(
        "Custo computacional (segundos)", "tab:custo", "llccc",
        ["Cenário & Configuração & " + " & ".join(n for _, n in AN.TEMPOS)],
        grupos_principal(rp, AN.TEMPOS, 3, traco_original=("tempo_smotenc_s",)),
        r"Média $\pm$ desvio-padrão de 5 repetições, com \texttt{n\_jobs} = "
        f"{int(n_jobs[0])}. O tempo de treino não inclui a geração das amostras sintéticas, "
        "registrada à parte. A predição corresponde à classificação de todo o conjunto de "
        f"teste ({milhar(n_teste['light'][0])} amostras no \\textit{{light}} e "
        f"{milhar(n_teste['heavy'][0])} no \\textit{{heavy}}).")


def tabela_08(rr):
    """Tabela de consulta com limiar 0,5 e limiar ajustado no treino — Seção 4.4."""
    grupos = []
    for cen in CENARIOS:
        g = []
        for var, nome in [("limiar05", "0,5"), ("limiarf1", "Ajustado no treino")]:
            i = idc(cen, "REF", var)
            g.append([CEN[cen], nome] + [mdp(rr, i, c, 4) for c, _ in AC.MET5])
        grupos.append(g)
    return tabela(
        "Linha de referência (tabela de consulta) no conjunto de teste", "tab:referencia",
        "llccccc", ["Cenário & Limiar & " + " & ".join(n for _, n in AC.MET5)], grupos,
        r"Média $\pm$ desvio-padrão de 5 repetições, com as mesmas divisões 70/30 e sementes "
        "do experimento principal.")


def tabela_09(rp, rr, rg):
    """Divisão aleatória (experimento principal) x validação por grupos — Seção 4.5."""
    met = [("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]
    grupos = []
    for cen in CENARIOS:
        g = []
        for m, b in CONFIGS + [("REF", "limiarf1")]:
            i = idc(cen, m, b)
            aleat = rr if m == "REF" else rp
            nome = AC.NOME_REF[b] if m == "REF" else rotulo(m, b)
            vals = []
            for c, _ in met:
                vals += [mdp(aleat, i, c, 4), mdp(rg, i, c, 4)]
            g.append([CEN[cen], nome] + vals)
        grupos.append(g)
    cab = " & ".join(f"{n} (aleat.) & {n} (grupo)" for _, n in met)
    return tabela(
        "Divisão aleatória (70/30) e validação por grupos de vetores", "tab:divisao_grupo",
        "llcccccc", ["Cenário & Configuração & " + cab], grupos,
        r"Média $\pm$ desvio-padrão de 5 repetições. Divisão aleatória: experimento "
        "principal. Por grupo: 3 dobras, sendo o grupo o vetor das 14 características; "
        "métricas sobre a predição fora da dobra de toda a base experimental.")


def tabela_10(rp, rb):
    """F1 por tratamento de balanceamento e limiar de decisão — Seção 4.6."""
    combos = [("original", rp), ("smotenc", rp), ("classweight", rb),
              ("original_limiar", rb), ("smotenc_limiar", rb), ("classweight_limiar", rb)]
    grupos, dps = [], []
    for cen in CENARIOS:
        g = []
        for m in ["DT", "RF"]:
            vals = []
            for var, r in combos:
                i = idc(cen, m, var)
                vals.append(f(r.loc[i, "f1_media"], 4))
                dps.append(r.loc[i, "f1_dp"])
            g.append([CEN[cen], AN.NOME_MODELO[m]] + vals)
        grupos.append(g)
    cw = r"\textit{class\_weight}"
    return tabela(
        r"\textit{F1-score} da classe de ataque por tratamento de balanceamento e limiar de "
        "decisão", "tab:balanceamento", "llcccccc",
        [r" & & \multicolumn{3}{c}{Limiar 0,5} & \multicolumn{3}{c}{Limiar ajustado no "
         r"treino}",
         f"Cenário & Modelo & Original & SMOTE-NC & {cw} & Original & SMOTE-NC & {cw}"],
        grupos,
        "Média de 5 repetições; os desvios-padrão foram de no máximo "
        f"{f(max(dps), 4)}. As colunas Original e SMOTE-NC com limiar 0,5 correspondem ao "
        "experimento principal.")


def tabela_11(rp):
    """Random Forest com as 14 características e sem o sld — Seção 4.8."""
    met = [("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]
    grupos = []
    for cen in CENARIOS:
        g = []
        for b in ["original", "smotenc"]:
            vals = []
            for c, _ in met:
                vals += [f(rp.loc[idc(cen, "RF", b), c + "_media"], 4),
                         f(rp.loc[idc(cen, "RF", b, "stateless_sem_sld"), c + "_media"], 4)]
            g.append([CEN[cen], AN.NOME_BAL[b]] + vals)
        grupos.append(g)
    sem = r"sem \texttt{sld}"
    cab = " & ".join(f"{n} (14) & {n} ({sem})" for _, n in met)
    return tabela(
        r"\textit{Random Forest} com e sem a característica \texttt{sld}", "tab:sem_sld",
        "llcccccc", ["Cenário & Configuração & " + cab], grupos,
        "Média de 5 repetições, com as mesmas divisões e sementes do experimento principal.")


# ---------------------------------------------------------------------------
# Figuras (reaproveitam as funções do 07, para ficarem idênticas às do TCC)
# ---------------------------------------------------------------------------
FIGURAS = {
    "referencia_light": "Figura 3 (Seção 4.4)",
    "referencia_heavy": "Figura 4 (Seção 4.4)",
    "importancia_light_RF": "Figura 5 (Seção 4.7)",
    "importancia_light_DT": "Figura 6 (Seção 4.7)",
}


def figuras(base, rp, rr, destino):
    os.makedirs(destino, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "figuras"))
        AC.analise_referencia(base, rp, rr, DEC, tmp, [])
        AC.analise_importancia(base, DEC, tmp, [])
        for nome in FIGURAS:
            for ext in ("pdf", "png"):
                origem = os.path.join(tmp, "figuras", f"{nome}.{ext}")
                if not os.path.exists(origem):
                    raise SystemExit(f"Figura {nome}.{ext} não foi gerada; confira os "
                                     "resultados dos scripts 03 e 06.")
                shutil.copy(origem, os.path.join(destino, f"{nome}.{ext}"))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="resultados",
                    help="pasta de resultados do experimento (padrão: resultados)")
    a = ap.parse_args()
    base = a.saida
    pc = os.path.join(base, "complementares")
    destino = os.path.join(base, "tcc")
    os.makedirs(destino, exist_ok=True)

    dp = AC.ler(os.path.join(base, "resultados_execucoes.csv"))
    if dp is None:
        raise SystemExit(f"Não encontrei {base}/resultados_execucoes.csv (script 03).")
    rp = AC.resumo(dp)
    rs = {}
    for exp in ["referencia", "grupo", "balanceamento"]:
        d = AC.ler(os.path.join(pc, exp, "resultados_execucoes.csv"))
        if d is None:
            raise SystemExit(f"Não encontrei {pc}/{exp}/resultados_execucoes.csv (script 06).")
        rs[exp] = AC.resumo(d)
    nota_repeticoes(rp, *rs.values())

    tabelas = [
        ("tabela_03_hiperparametros", tabela_03(base)),
        ("tabela_04_vetores_por_classe", tabela_04(base)),
        ("tabela_05_metricas", tabela_05(rp)),
        ("tabela_06_matriz_confusao", tabela_06(rp)),
        ("tabela_07_custo", tabela_07(rp, dp)),
        ("tabela_08_referencia", tabela_08(rs["referencia"])),
        ("tabela_09_divisao_grupo", tabela_09(rp, rs["referencia"], rs["grupo"])),
        ("tabela_10_balanceamento", tabela_10(rp, rs["balanceamento"])),
        ("tabela_11_sem_sld", tabela_11(rp)),
    ]
    cabecalho = "% Gerado por 08_tabelas_figuras_tcc.py. Requer \\usepackage{graphicx}.\n"
    for nome, conteudo in tabelas:
        with open(os.path.join(destino, nome + ".tex"), "w", encoding="utf-8") as fh:
            fh.write(cabecalho + conteudo)
    with open(os.path.join(destino, "capitulo4_tabelas.tex"), "w", encoding="utf-8") as fh:
        fh.write(cabecalho + "\n".join(c for _, c in tabelas))

    figuras(base, rp, rs["referencia"], os.path.join(destino, "figuras"))

    print(f"Gerado em {os.path.abspath(destino)}:")
    for nome, _ in tabelas:
        print(f"  {nome}.tex")
    for nome, desc in FIGURAS.items():
        print(f"  figuras/{nome}.pdf/.png  ({desc})")


if __name__ == "__main__":
    main()
