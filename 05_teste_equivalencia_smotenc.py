"""Teste de equivalência: SMOTENCCronometrado (comum.py) x SMOTENC original do imblearn.

Por que existe: o SMOTENC do imblearn não coube na memória (11,6 GB, Seção 4.1)
com as ~85 mil categorias de sld e longest_word. O comum.py usa uma versão
reescrita, e este script comprova que ela gera exatamente as mesmas amostras,
ou seja, que o método continua sendo o SMOTE-NC da Seção 3.3.

Usa uma subamostra estratificada de cada cenário, pequena o bastante para o SMOTENC
original caber na memória. Monta o mesmo codificador do pipeline, aplica os dois
amostradores com os mesmos parâmetros e a mesma semente e compara as saídas elemento
a elemento. Se tudo for idêntico, a substituição não altera o experimento; só o
consumo de memória e o tempo mudam.

Uso:
  python 05_teste_equivalencia_smotenc.py --dados dados/CICBellEXFDNS2021.zip
  python 05_teste_equivalencia_smotenc.py --dados dados/CICBellEXFDNS2021.zip --fracao 0.03 --sementes 0 1 2 3 4

O relatório é salvo em <saida>/verificacao/equivalencia_smotenc.txt.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import comum as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--fracao", type=float, default=0.02,
                    help="fração de cada cenário usada no teste (padrão 0,02; aumente com cuidado: "
                         "o SMOTENC original gasta muita memória)")
    ap.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    import imblearn
    from imblearn.over_sampling import SMOTENC

    linhas = [f"Teste de equivalência SMOTE-NC — {C.agora()}",
              f"imblearn {imblearn.__version__} | fração {a.fracao} | sementes {a.sementes}", ""]
    arquivos = C.carregar_arquivos(a.dados)
    tudo_ok = True
    for conjunto in [C.CONJUNTO_PRINCIPAL, C.CONJUNTO_SEM_SLD]:
        for cen in C.CENARIOS:
            df = C.montar_cenario(arquivos, cen)
            _, idx = train_test_split(np.arange(len(df)), test_size=a.fracao,
                                      stratify=df.classe.values,
                                      random_state=C.SEMENTE_SUBAMOSTRA_EQUIVALENCIA)
            sub = df.iloc[np.sort(idx)]
            cat, num = C.colunas_do_conjunto(conjunto)
            for s in a.sementes:
                pipe = C.montar_pipeline("DT", "smotenc", conjunto, semente=s, n_jobs=1)
                X = pipe.named_steps["codificacao"].fit_transform(sub[cat + num], sub.classe.values)
                y = sub.classe.values.astype(int)
                novo = pipe.named_steps["smotenc"]
                params = novo.get_params(deep=False)
                params["k_neighbors"] = clone(params["k_neighbors"])
                orig = SMOTENC(**params)

                t0 = time.perf_counter()
                Xo, yo = orig.fit_resample(X, y)
                t_o = time.perf_counter() - t0
                t0 = time.perf_counter()
                Xn, yn = novo.fit_resample(X, y)
                t_n = time.perf_counter() - t0

                Xo, Xn = np.asarray(Xo), np.asarray(Xn)
                mesma_forma = Xo.shape == Xn.shape and yo.shape == yn.shape
                igual_y = mesma_forma and np.array_equal(yo, yn)
                igual_x = mesma_forma and np.array_equal(Xo, Xn)
                dif = float(np.max(np.abs(Xo - Xn))) if mesma_forma else float("nan")
                ok = igual_x and igual_y
                tudo_ok &= ok
                linhas.append(
                    f"{'OK  ' if ok else 'FALHA'} {cen:5s} {conjunto:18s} semente {s}: "
                    f"entrada {len(y)} ({int(y.sum())} ataque) -> {len(yn)} | "
                    f"X idêntico={igual_x} y idêntico={igual_y} dif_máx={dif:.3g} | "
                    f"tempo original {t_o:.2f}s, novo {t_n:.2f}s")
                print(linhas[-1], flush=True)
    linhas += ["", "RESULTADO: " + ("todas as saídas são idênticas ao SMOTENC do imblearn."
                                    if tudo_ok else "há diferenças — NÃO use a versão nova.")]
    destino = os.path.join(a.saida, "verificacao", "equivalencia_smotenc.txt")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    print("\n" + linhas[-1] + f"\nRelatório: {destino}")
    raise SystemExit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
