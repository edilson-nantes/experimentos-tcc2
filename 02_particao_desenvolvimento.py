"""Script 2 — partição de desenvolvimento (Seções 3.3 e 3.4 do TCC 2).

Separa 15% de cada cenário, de forma estratificada e com semente fixa
(comum.SEMENTE_PARTICAO_DEV = 42), para o ajuste de hiperparâmetros. Essa
partição é separada uma única vez, fica fixa em todo o experimento e não
participa de nenhuma avaliação. Os 85% restantes formam a base experimental.
Os tamanhos resultantes são os informados na Seção 4.1 (48.578 amostras no
light e 98.166 no heavy).

Os índices são salvos em arquivo, com um hash dos dados, e NÃO são
sobrescritos sem --sobrescrever. Os scripts 03 e 06 conferem esse hash antes
de usar a partição.

Uso:
  python 02_particao_desenvolvimento.py --dados dados/CICBellEXFDNS2021.zip --saida resultados
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import comum as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--sobrescrever", action="store_true")
    a = ap.parse_args()

    pasta = os.path.join(a.saida, "particoes")
    os.makedirs(pasta, exist_ok=True)
    arquivos = C.carregar_arquivos(a.dados)
    for cen in C.CENARIOS:
        destino = os.path.join(pasta, f"dev_{cen}.csv")
        if os.path.exists(destino) and not a.sobrescrever:
            print(f"[{cen}] partição já existe em {destino} — mantida (use --sobrescrever).")
            continue
        df = C.montar_cenario(arquivos, cen)
        idx = np.arange(len(df))
        _, idx_dev = train_test_split(idx, test_size=C.FRACAO_DEV, stratify=df.classe.values,
                                      random_state=C.SEMENTE_PARTICAO_DEV)
        idx_dev = np.sort(idx_dev)
        pd.DataFrame({
            "indice": idx_dev,
            "arquivo": df.arquivo.values[idx_dev],
            "linha_no_arquivo": df.linha.values[idx_dev],
            "classe": df.classe.values[idx_dev],
        }).to_csv(destino, index=False)
        meta = {
            "cenario": cen,
            "n_total": int(len(df)),
            "n_ataque_total": int(df.classe.sum()),
            "n_dev": int(len(idx_dev)),
            "n_ataque_dev": int(df.classe.values[idx_dev].sum()),
            "n_base_experimental": int(len(df) - len(idx_dev)),
            "fracao_dev": C.FRACAO_DEV,
            "semente": C.SEMENTE_PARTICAO_DEV,
            "estratificada": True,
            "hash_dados": C.hash_dados(df),
            "criado_em": C.agora(),
            "versoes": C.versoes_bibliotecas(),
        }
        C.gravar_json(os.path.join(pasta, f"dev_{cen}.json"), meta)
        print(f"[{cen}] dev={meta['n_dev']} (ataque {meta['n_ataque_dev']}), "
              f"base experimental={meta['n_base_experimental']} -> {destino}")


if __name__ == "__main__":
    main()
