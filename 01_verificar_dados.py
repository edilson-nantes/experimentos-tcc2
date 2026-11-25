"""Script 1 — verificação dos dados (Seção 4.2 do TCC 2).

Roda antes de qualquer experimento. Não altera os dados usados (a metodologia
não prevê remoção de duplicatas); serve para interpretar os resultados.

Gera um relatório curto em texto com:
  - arquivos encontrados, colunas e tipos (quais são textuais);
  - contagens por cenário e classe, comparadas com a Tabela 1 (Seção 3.2);
  - valores ausentes (leitura padrão do pandas x leitura correta): os 31
    "ausentes" de longest_word são as palavras "nan"/"null" (Seção 4.2);
  - duplicatas e vetores de características que aparecem nas duas classes,
    a base da Tabela 4 e do teto de desempenho medido na Seção 4.4;
  - tabulação cruzada sld x classe (e longest_word x classe), citada nas
    Seções 4.2 e 4.8.

Saídas em <saida>/verificacao/:
  relatorio_verificacao.txt, resumo_vetores.csv (números da Tabela 4,
  lidos pelo 08_tabelas_figuras_tcc.py), tabulacao_*.csv e
  vetores_nas_duas_classes_*.csv.

Uso:
  python 01_verificar_dados.py --dados dados/CICBellEXFDNS2021.zip --saida resultados
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pandas as pd

import comum as C


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def bloco_tipos(fontes, R):
    R.append("1. ARQUIVOS, COLUNAS E TIPOS")
    R.append("-" * 70)
    textuais_detectadas = set()
    problemas = []
    for nome in sorted(C.ARQUIVOS_ESPERADOS):
        origem, conteudo = fontes[nome]
        bruto = pd.read_csv(io.BytesIO(conteudo), low_memory=False)  # leitura padrão
        cens, classe = C.ARQUIVOS_ESPERADOS[nome]
        cols = [c for c in bruto.columns if c != C.COLUNA_TEMPO]
        if cols != C.CARACTERISTICAS:
            problemas.append(f"{nome}: colunas diferentes do esperado: {cols}")
        for c in C.CARACTERISTICAS:
            if c in bruto.columns:
                conv = pd.to_numeric(bruto[c], errors="coerce")
                if (conv.isna() & bruto[c].notna()).any():
                    textuais_detectadas.add(c)
        R.append(f"  {nome:48s} {len(bruto):>8d} linhas  cenário={'+'.join(cens):11s} "
                 f"classe={'ataque' if classe else 'benigno'}")
    R.append("")
    R.append(f"  Colunas além de '{C.COLUNA_TEMPO}': {len(C.CARACTERISTICAS)} — "
             + ", ".join(C.CARACTERISTICAS))
    R.append(f"  Colunas textuais detectadas: {sorted(textuais_detectadas)}")
    esperado = sorted(C.COLUNAS_TEXTUAIS_ESPERADAS)
    R.append(f"  Esperado: {esperado} -> "
             + ("OK" if sorted(textuais_detectadas) == esperado else "DIVERGENTE"))
    for p in problemas:
        R.append("  PROBLEMA: " + p)
    R.append("  Rótulo: não há coluna de classe nos CSVs; a classe vem da pasta/arquivo")
    R.append("  (Attacks = ataque, Benign = benigno). Cada arquivo tem uma única classe.")
    R.append("")
    return sorted(textuais_detectadas)


def bloco_ausentes(fontes, arquivos, R):
    R.append("3. VALORES AUSENTES")
    R.append("-" * 70)
    tot_padrao = {}
    for nome in sorted(C.ARQUIVOS_ESPERADOS):
        bruto = pd.read_csv(io.BytesIO(fontes[nome][1]), low_memory=False)
        for c, n in bruto.isna().sum().items():
            if n:
                tot_padrao[c] = tot_padrao.get(c, 0) + int(n)
    tot_correto = {}
    for nome, d in arquivos.items():
        for c, n in d.isna().sum().items():
            if n:
                tot_correto[c] = tot_correto.get(c, 0) + int(n)
    R.append(f"  Leitura padrão do pandas (pd.read_csv sem opções): {tot_padrao or 'nenhum'}")
    R.append(f"  Leitura usada no experimento (keep_default_na=False): {tot_correto or 'nenhum'}")
    if tot_padrao and not tot_correto:
        R.append("  Os 'ausentes' da leitura padrão são as palavras literais 'nan'/'null'")
        R.append("  em longest_word (ex.: domínios nanit, nullpk), que o pandas converte")
        R.append("  em NaN. Não são valores ausentes de verdade; o experimento os mantém")
        R.append("  como texto. A substituição por zero (Cap. 3) continua no código, mas")
        R.append("  não altera nenhuma linha.")
    R.append("")


def bloco_contagens(dfs, R):
    R.append("2. CONTAGENS POR CENÁRIO E CLASSE (comparação com a Tabela 1)")
    R.append("-" * 70)
    R.append(f"  {'Cenário':8s} {'Benigno':>9s} {'Ataque':>9s} {'Total':>9s} {'Ataque%':>8s}  Tabela 1")
    for cen, df in dfs.items():
        b = int((df.classe == 0).sum()); a = int((df.classe == 1).sum())
        e = C.TABELA1[cen]
        ok = "OK" if (b, a) == (e["benigno"], e["ataque"]) else \
            f"DIVERGENTE (esperado {e['benigno']}/{e['ataque']})"
        R.append(f"  {cen:8s} {b:>9d} {a:>9d} {a + b:>9d} {pct(a, a + b):>7.1f}%  {ok}")
    R.append("  Obs.: benign_1 e benign_2 (Benign.zip na raiz da pasta CSV) entram nos")
    R.append("  dois cenários; só assim as contagens coincidem com a Tabela 1.")
    R.append("")
    R.append("  Linhas de ataque por tipo de arquivo:")
    for cen, df in dfs.items():
        cont = df[df.classe == 1].arquivo.str.extract(r"-(?:light|heavy)_(\w+)\.pcap")[0].value_counts()
        R.append(f"    {cen}: " + ", ".join(f"{k}={v}" for k, v in cont.sort_index().items()))
    R.append("")


def bloco_duplicatas(dfs, arquivos, saida, R):
    """Vetores das 14 características por classe (Tabela 4 da Seção 4.2).

    Se um mesmo vetor aparece com as duas classes, nenhum classificador que
    use só essas 14 características acerta os dois registros. É isso que
    impõe o teto de desempenho medido na Seção 4.4.
    """
    R.append("4. DUPLICATAS E VETORES PRESENTES NAS DUAS CLASSES")
    R.append("-" * 70)
    R.append("  'Vetor' = valores das 14 características (sem timestamp). Pela metodologia,")
    R.append("  duplicatas NÃO são removidas; os números abaixo servem para interpretação.")
    resumo_teto = {}
    tabela4 = []  # uma linha por cenário -> resumo_vetores.csv (Tabela 4)
    for cen, df in dfs.items():
        n = len(df)
        chave = pd.util.hash_pandas_object(df[C.CARACTERISTICAS], index=False)
        # duplicatas incluindo timestamp
        partes = []
        for nome in sorted(arquivos):
            if cen in C.ARQUIVOS_ESPERADOS[nome][0]:
                partes.append(arquivos[nome])
        com_ts = pd.concat(partes, ignore_index=True)
        dup_ts = int(com_ts.duplicated().sum())
        dup = int(chave.duplicated().sum())
        g = pd.DataFrame({"k": chave.values, "y": df.classe.values})
        por_vetor = g.groupby("k").y.agg(["sum", "count"])
        por_vetor["ben"] = por_vetor["count"] - por_vetor["sum"]
        conflito = por_vetor[(por_vetor["sum"] > 0) & (por_vetor["ben"] > 0)]
        n_at = int(df.classe.sum()); n_be = n - n_at
        at_conf = int(conflito["sum"].sum()); be_conf = int(conflito["ben"].sum())
        R.append(f"  [{cen}] linhas: {n}")
        R.append(f"    duplicatas exatas incluindo timestamp: {dup_ts} ({pct(dup_ts, n):.1f}%)")
        R.append(f"    duplicatas das 14 características:      {dup} ({pct(dup, n):.1f}%)")
        R.append(f"    vetores distintos: benigno={int((por_vetor.ben > 0).sum())}, "
                 f"ataque={int((por_vetor['sum'] > 0).sum())}, nas duas classes={len(conflito)}")
        R.append(f"    linhas de ATAQUE cujo vetor também aparece como benigno: "
                 f"{at_conf} de {n_at} ({pct(at_conf, n_at):.1f}%)")
        R.append(f"    linhas BENIGNAS cujo vetor também aparece como ataque:   "
                 f"{be_conf} de {n_be} ({pct(be_conf, n_be):.1f}%)")
        tabela4.append({
            "cenario": cen, "n_benigno": n_be, "n_ataque": n_at,
            "n_duplicatas_14": dup, "pct_duplicatas_14": pct(dup, n),
            "vetores_benigno": int((por_vetor.ben > 0).sum()),
            "vetores_ataque": int((por_vetor["sum"] > 0).sum()),
            "vetores_nas_duas_classes": len(conflito),
            "ataque_com_vetor_benigno": at_conf, "pct_ataque_com_vetor_benigno": pct(at_conf, n_at),
            "benigno_com_vetor_ataque": be_conf, "pct_benigno_com_vetor_ataque": pct(be_conf, n_be),
        })
        # Teto: classificador ótimo que vê só as 14 características prevê, para
        # cada vetor, a classe majoritária daquele vetor (avaliação nos próprios dados).
        prev_ataque = por_vetor["sum"] > por_vetor["ben"]
        tp = int(por_vetor.loc[prev_ataque, "sum"].sum())
        fp = int(por_vetor.loc[prev_ataque, "ben"].sum())
        fn = int(por_vetor.loc[~prev_ataque, "sum"].sum())
        tn = n - tp - fp - fn
        P = tp / (tp + fp) if tp + fp else 0.0
        Rc = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
        resumo_teto[cen] = (P, Rc, F1, (tp + tn) / n)
        R.append(f"    TETO com as 14 características (regra 'classe majoritária de cada")
        R.append(f"    vetor', nos próprios dados): precisão={P:.4f} recall={Rc:.4f} "
                 f"F1={F1:.4f} acurácia={(tp + tn) / n:.4f}")
        conflito.sort_values("sum", ascending=False).rename(
            columns={"sum": "n_ataque", "ben": "n_benigno", "count": "n_total"}
        ).head(200).to_csv(os.path.join(saida, f"vetores_nas_duas_classes_{cen}.csv"))
        # exemplos legíveis dos vetores de ataque mais frequentes
        ex = df[df.classe == 1][C.CARACTERISTICAS].value_counts().head(5)
        R.append("    vetores de ataque mais frequentes (contagem):")
        for vals, cnt in ex.items():
            d = dict(zip(C.CARACTERISTICAS, vals))
            R.append(f"      {cnt:>7d}  sld={d['sld']!r} longest_word={d['longest_word']!r} "
                     f"FQDN_count={d['FQDN_count']} subdomain_length={d['subdomain_length']} "
                     f"entropy={float(d['entropy']):.3f}")
    R.append("")
    pd.DataFrame(tabela4).to_csv(os.path.join(saida, "resumo_vetores.csv"), index=False)
    return resumo_teto


def bloco_textual(dfs, coluna, saida, R, numero):
    R.append(f"{numero}. TABULAÇÃO CRUZADA {coluna} x CLASSE")
    R.append("-" * 70)
    for cen, df in dfs.items():
        ct = pd.crosstab(df[coluna], df.classe).rename(columns={0: "benigno", 1: "ataque"})
        for col in ["benigno", "ataque"]:
            if col not in ct:
                ct[col] = 0
        ct["total"] = ct.benigno + ct.ataque
        ct = ct.sort_values(["ataque", "total"], ascending=False)
        ct.to_csv(os.path.join(saida, f"tabulacao_{coluna}_classe_{cen}.csv"))
        so_b = ct[(ct.benigno > 0) & (ct.ataque == 0)]
        so_a = ct[(ct.ataque > 0) & (ct.benigno == 0)]
        ambos = ct[(ct.ataque > 0) & (ct.benigno > 0)]
        n_at = int(ct.ataque.sum())
        R.append(f"  [{cen}] valores distintos de {coluna}: {len(ct)}")
        R.append(f"    só benigno: {len(so_b)} | só ataque: {len(so_a)} | nas duas: {len(ambos)}")
        R.append(f"    linhas de ataque com {coluna} exclusivo de ataque: {int(so_a.ataque.sum())} "
                 f"de {n_at} ({pct(so_a.ataque.sum(), n_at):.2f}%)")
        maj = ct.ataque > ct.benigno
        tp = int(ct.ataque[maj].sum()); fp = int(ct.benigno[maj].sum())
        fn = int(ct.ataque[~maj].sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        Rc = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
        R.append(f"    regra só com {coluna} (classe majoritária de cada valor): "
                 f"precisão={P:.4f} recall={Rc:.4f} F1={F1:.4f}")
        R.append(f"    {coluna} mais frequentes no ataque (benigno / ataque):")
        for v, linha in ct.head(10).iterrows():
            R.append(f"      {str(v)[:40]:40s} {int(linha.benigno):>8d} / {int(linha.ataque):>8d}")
    R.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True,
                    help="CICBellEXFDNS2021.zip ou a pasta extraída")
    ap.add_argument("--saida", default="resultados")
    a = ap.parse_args()
    saida = os.path.join(a.saida, "verificacao")
    os.makedirs(saida, exist_ok=True)

    fontes = C.bytes_dos_arquivos(a.dados)
    faltando = sorted(set(C.ARQUIVOS_ESPERADOS) - set(fontes))
    if faltando:
        raise SystemExit("Arquivos ausentes: " + ", ".join(faltando))
    arquivos = {n: C.ler_csv_stateless(fontes[n][1]) for n in sorted(C.ARQUIVOS_ESPERADOS)}
    dfs = {cen: C.montar_cenario(arquivos, cen) for cen in C.CENARIOS}

    R = ["RELATÓRIO DE VERIFICAÇÃO DOS DADOS — CIC-Bell-DNS-EXF-2021 (stateless)",
         f"Gerado em {C.agora()} | dados: {os.path.basename(a.dados)}", ""]
    bloco_tipos(fontes, R)
    bloco_contagens(dfs, R)
    bloco_ausentes(fontes, arquivos, R)
    teto = bloco_duplicatas(dfs, arquivos, saida, R)
    bloco_textual(dfs, "sld", saida, R, 5)
    bloco_textual(dfs, "longest_word", saida, R, 6)

    R.append("7. TIPOS APÓS A LEITURA USADA NO EXPERIMENTO")
    R.append("-" * 70)
    for c, t in dfs["light"][C.CARACTERISTICAS].dtypes.items():
        R.append(f"  {c:18s} {t}")
    R.append("")
    R.append("8. IMPRESSÃO DIGITAL DOS DADOS (SHA-256 das 14 características + classe)")
    R.append("-" * 70)
    for cen, df in dfs.items():
        R.append(f"  {cen}: {C.hash_dados(df)}")
    R.append("")
    R.append("RESUMO")
    R.append("-" * 70)
    for cen, (P, Rc, F1, acc) in teto.items():
        R.append(f"  {cen}: F1 máximo atingível com as 14 características ≈ {F1:.3f} "
                 f"(precisão {P:.3f}, recall {Rc:.3f})")
    if min(v[2] for v in teto.values()) < 0.9:
        R.append("  ALERTA: há vetores idênticos rotulados como ataque e como benigno em")
        R.append("  grande proporção. Nenhum modelo que use só estas 14 características")
        R.append("  passa do teto acima. Discuta com o orientador antes da redação.")

    texto = "\n".join(R) + "\n"
    caminho = os.path.join(saida, "relatorio_verificacao.txt")
    with open(caminho, "w", encoding="utf-8") as fh:
        fh.write(texto)
    print(texto)
    print(f"Relatório salvo em {caminho}")


if __name__ == "__main__":
    main()
