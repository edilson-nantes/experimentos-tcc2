"""Script 3 — experimento principal (procedimento das Seções 3.3 a 3.6 do TCC 2).

Para cada uma das 8 configurações da Tabela 2 (cenário x modelo x balanceamento):
  1. Grid Search uma única vez, na partição de desenvolvimento (15%),
     StratifiedKFold k=5, maximizando o F1 da classe de ataque (Seção 3.4).
     Os hiperparâmetros escolhidos são salvos em JSON e reaproveitados nas 5
     repetições e nos experimentos complementares -> Tabela 3 (Seção 4.1).
  2. Para cada semente (0 a 4): divisão estratificada 70/30 da base
     experimental (85%), pipeline OrdinalEncoder -> [SMOTE-NC] -> classificador
     ajustado só no treino, avaliação no teste com a proporção original
     (Seções 3.3, 3.5 e 3.6) -> Tabelas 5, 6 e 7 (Seção 4.3).
  3. Cada execução vira uma linha no CSV assim que termina. Ao reiniciar, o
     script pula o que já está no CSV.

Com --sem-sld, acrescenta o Random Forest sem a coluna sld, com novo Grid
Search (Seção 4.8, Tabela 11).

Uso típico:
  python 03_experimento.py --dados CICBellEXFDNS2021.zip --teste-rapido   # 2-5 min
  python 03_experimento.py --dados CICBellEXFDNS2021.zip --piloto         # 1 config, 1 semente
  python 03_experimento.py --dados CICBellEXFDNS2021.zip                  # experimento completo
  python 03_experimento.py --dados CICBellEXFDNS2021.zip --sem-sld        # + RF sem a coluna sld
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, make_scorer, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import comum as C

# Uma linha por (configuração, semente). Métricas da classe de ataque (classe
# positiva) no conjunto de teste, matriz de confusão e tempos (Seção 3.6).
COLUNAS_CSV = [
    "id_config", "cenario", "modelo", "balanceamento", "conjunto_caracteristicas", "semente",
    "precisao", "recall", "f1", "acuracia", "taxa_falsos_positivos", "roc_auc", "pr_auc",
    "vp", "vn", "fp", "fn",
    "tempo_treino_s", "tempo_predicao_s", "tempo_smotenc_s", "tempo_predict_proba_s",
    "tempo_fit_total_s",
    "n_treino", "n_treino_ataque", "n_treino_apos_smotenc", "n_teste", "n_teste_ataque",
    "hiperparametros", "n_jobs", "hostname", "inicio", "fim",
]

# Grades reduzidas, usadas só no --teste-rapido (não entram no TCC)
GRADE_TESTE_RF = {"n_estimators": [10, 20], "max_depth": [5, None], "min_samples_split": [2]}
GRADE_TESTE_DT = {"max_depth": [5, None], "min_samples_split": [2]}


# ---------------------------------------------------------------------------
def lista_configs(sem_sld: bool) -> list[str]:
    ids = [C.id_config(c, m, b, C.CONJUNTO_PRINCIPAL)
           for c in C.CENARIOS for m in C.MODELOS for b in C.BALANCEAMENTOS]
    if sem_sld:
        ids += [C.id_config(c, "RF", b, C.CONJUNTO_SEM_SLD)
                for c in C.CENARIOS for b in C.BALANCEAMENTOS]
    return ids


def verificar_ambiente(base: str, n_jobs: int, forcar: bool, log) -> dict:
    """Registra versões, CPU e n_jobs em ambiente.json (dados da Seção 4.1).

    Recusa n_jobs diferente do já registrado: os tempos da Tabela 7 só são
    comparáveis se todas as execuções usarem o mesmo número de núcleos
    (Seção 3.5).
    """
    amb = {"n_jobs": n_jobs, "versoes": C.versoes_bibliotecas(),
           "hardware": C.info_hardware(), "registrado_em": C.agora(),
           "comando": " ".join(sys.argv)}
    caminho = os.path.join(base, "ambiente.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as fh:
            ant = json.load(fh)
        if ant["n_jobs"] != n_jobs and not forcar:
            log.error("n_jobs=%s difere do usado antes (%s, em %s). Os tempos não seriam "
                      "comparáveis. Use --n-jobs %s (ou --forcar-ambiente).",
                      n_jobs, ant["n_jobs"], caminho, ant["n_jobs"])
            raise SystemExit(2)
        if ant["versoes"] != amb["versoes"]:
            log.warning("Versões de bibliotecas mudaram desde %s: antes=%s agora=%s",
                        ant["registrado_em"], ant["versoes"], amb["versoes"])
        if ant["hardware"].get("cpu") != amb["hardware"].get("cpu"):
            log.warning("CPU diferente da registrada antes: %s -> %s",
                        ant["hardware"].get("cpu"), amb["hardware"].get("cpu"))
    else:
        C.gravar_json(caminho, amb)
    with open(os.path.join(base, "ambiente_historico.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: v for k, v in amb.items() if k != "hardware"}
                            | {"cpu": amb["hardware"].get("cpu")}, ensure_ascii=False) + "\n")
    log.info("Ambiente: python %s | sklearn %s | imblearn %s | pandas %s | numpy %s",
             amb["versoes"]["python"], amb["versoes"]["sklearn"], amb["versoes"]["imblearn"],
             amb["versoes"]["pandas"], amb["versoes"]["numpy"])
    log.info("Hardware: %s | %s núcleos físicos / %s lógicos | %s GB RAM | n_jobs=%s",
             amb["hardware"].get("cpu"), amb["hardware"].get("nucleos_fisicos"),
             amb["hardware"].get("nucleos_logicos"), amb["hardware"].get("ram_gb"), n_jobs)
    return amb


def preparar_cenario(arquivos, cen: str, base: str, teste_rapido: bool, log):
    """Retorna (df, mascara_dev) do cenário.

    mascara_dev marca a partição de desenvolvimento de 15% (Seção 3.3), lida
    do arquivo gravado pelo script 02 e conferida pelo hash dos dados.
    """
    df = C.montar_cenario(arquivos, cen)
    if teste_rapido:
        _, idx = train_test_split(np.arange(len(df)), test_size=0.04,
                                  stratify=df.classe.values,
                                  random_state=C.SEMENTE_TESTE_RAPIDO)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
        _, idx_dev = train_test_split(np.arange(len(df)), test_size=C.FRACAO_DEV,
                                      stratify=df.classe.values, random_state=C.SEMENTE_PARTICAO_DEV)
    else:
        caminho = os.path.join(base, "particoes", f"dev_{cen}.csv")
        meta_c = os.path.join(base, "particoes", f"dev_{cen}.json")
        if not os.path.exists(caminho):
            log.error("Partição de desenvolvimento não encontrada (%s). Rode antes o "
                      "02_particao_desenvolvimento.py com a mesma --saida.", caminho)
            raise SystemExit(2)
        with open(meta_c, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["n_total"] != len(df) or meta["hash_dados"] != C.hash_dados(df):
            log.error("[%s] Os dados não coincidem com os usados para criar a partição "
                      "(n=%s vs %s ou hash diferente).", cen, len(df), meta["n_total"])
            raise SystemExit(2)
        part = pd.read_csv(caminho)
        idx_dev = part.indice.values
        ok = (df.arquivo.values[idx_dev] == part.arquivo.values).all() and \
             (df.linha.values[idx_dev] == part.linha_no_arquivo.values).all()
        if not ok:
            log.error("[%s] Índices da partição não correspondem às linhas esperadas.", cen)
            raise SystemExit(2)
    mascara = np.zeros(len(df), dtype=bool)
    mascara[idx_dev] = True
    log.info("[%s] total=%d | dev=%d (ataque %d) | base experimental=%d (ataque %d)",
             cen, len(df), mascara.sum(), df.classe.values[mascara].sum(),
             (~mascara).sum(), df.classe.values[~mascara].sum())
    return df, mascara


def ler_feitos(caminho_csv: str) -> set:
    if not os.path.exists(caminho_csv):
        return set()
    d = pd.read_csv(caminho_csv)
    return set(zip(d.id_config, d.semente.astype(int)))


def anexar_linha(caminho_csv: str, linha: dict) -> None:
    novo = not os.path.exists(caminho_csv)
    if not novo:
        with open(caminho_csv, encoding="utf-8") as fh:
            cab = fh.readline().strip().split(",")
        if cab != COLUNAS_CSV:
            raise RuntimeError(f"Cabeçalho de {caminho_csv} difere do esperado.")
    with open(caminho_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUNAS_CSV)
        if novo:
            w.writeheader()
        w.writerow(linha)
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
def obter_hiperparametros(idc, df, mascara, pasta_hp, n_jobs_grid, teste_rapido, log):
    """Grid Search da configuração `idc` (Seção 3.4) ou leitura do JSON já salvo.

    Roda só na partição de desenvolvimento, com validação cruzada estratificada
    de 5 dobras e F1 da classe de ataque como critério. Como o SMOTE-NC está
    dentro do pipeline, as amostras sintéticas são geradas só nas dobras de
    treino (Seção 3.3). Em caso de empate, vence a primeira combinação da
    grade; o número de empates é registrado (coluna "Empates" da Tabela 3).
    """
    caminho = os.path.join(pasta_hp, f"{idc}.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as fh:
            hp = json.load(fh)
        log.info("[%s] hiperparâmetros já ajustados: %s (F1 cv=%.4f)", idc,
                 hp["melhores_hiperparametros"], hp["f1_cv_media"])
        return hp["melhores_hiperparametros"]

    p = C.separar_id(idc)
    cat, num = C.colunas_do_conjunto(p["conjunto"])
    X = df.loc[mascara, cat + num]
    y = df.classe.values[mascara]
    if p["modelo"] == "RF":
        grade = GRADE_TESTE_RF if teste_rapido else C.GRADE_RF
    else:
        grade = GRADE_TESTE_DT if teste_rapido else C.GRADE_DT
    # n_jobs_grid paraleliza as combinações da grade; dentro de cada ajuste usa-se 1
    # núcleo. Cada processo paralelo tem sua própria cópia dos dados e do modelo, então
    # a memória cresce com n_jobs_grid. Não afeta os tempos das execuções principais.
    pipe = C.montar_pipeline(p["modelo"], p["balanceamento"], p["conjunto"],
                             semente=C.SEMENTE_GRID, n_jobs=1)
    cv = StratifiedKFold(n_splits=C.K_FOLDS, shuffle=True, random_state=C.SEMENTE_GRID)
    gs = GridSearchCV(pipe, {f"classificador__{k}": v for k, v in grade.items()},
                      scoring=make_scorer(f1_score, pos_label=1, zero_division=0),
                      cv=cv, n_jobs=n_jobs_grid, refit=False, error_score="raise")
    n_comb = int(np.prod([len(v) for v in grade.values()]))
    log.info("[%s] Grid Search: %d combinações x %d folds em %d amostras (ataque %d), "
             "n_jobs_grid=%d...", idc, n_comb, C.K_FOLDS, len(y), int(y.sum()), n_jobs_grid)
    t0 = time.perf_counter()
    gs.fit(X, y)
    t_grid = time.perf_counter() - t0
    res = pd.DataFrame(gs.cv_results_)
    i = int(np.argmin(res.rank_test_score.values))  # empate -> primeira combinação da grade
    melhores = {k.replace("classificador__", ""): v for k, v in res.params[i].items()}
    n_empate = int((res.rank_test_score == 1).sum())
    pasta_cv = os.path.join(pasta_hp, "cv_results")
    os.makedirs(pasta_cv, exist_ok=True)
    res.drop(columns=["params"]).to_csv(os.path.join(pasta_cv, f"{idc}.csv"), index=False)
    C.gravar_json(caminho, {
        "id_config": idc, **p,
        "melhores_hiperparametros": melhores,
        "f1_cv_media": float(res.mean_test_score[i]),
        "f1_cv_dp": float(res.std_test_score[i]),
        "combinacoes_empatadas_no_topo": n_empate,
        "grade": grade, "k_folds": C.K_FOLDS, "semente_cv": C.SEMENTE_GRID,
        "n_dev": int(len(y)), "n_dev_ataque": int(y.sum()),
        "tempo_grid_s": t_grid, "n_jobs_grid": n_jobs_grid,
        "teste_rapido": teste_rapido, "criado_em": C.agora(),
        "versoes": C.versoes_bibliotecas(),
    })
    log.info("[%s] melhores=%s | F1 cv=%.4f ± %.4f | empates no topo=%d | %.1f s",
             idc, melhores, res.mean_test_score[i], res.std_test_score[i], n_empate, t_grid)
    if n_empate > 1:
        log.warning("[%s] %d combinações empataram no melhor F1; escolhida a primeira da grade.",
                    idc, n_empate)
    return melhores


def executar(idc, hp, df, mascara, semente, n_jobs, pasta, log):
    """Uma repetição (semente) de uma configuração — passos a) a d) da Seção 3.5.

    a) divisão estratificada 70/30 da base experimental com a semente;
    b) codificação e, se previsto, SMOTE-NC ajustados só no treino;
    c) treino com os hiperparâmetros do Grid Search (tempo cronometrado);
    d) predição no teste, que mantém a proporção original de cada cenário, e
       cálculo das métricas da Seção 3.6.
    A agregação em média e desvio-padrão (passo e) é feita pelo 04_analise.py.
    """
    p = C.separar_id(idc)
    cat, num = C.colunas_do_conjunto(p["conjunto"])
    base = df.loc[~mascara].reset_index(drop=True)
    X = base[cat + num]
    y = base.classe.values.astype(int)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=C.FRACAO_TESTE,
                                      stratify=y, random_state=semente)
    Xtr, ytr, Xte, yte = X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te], y[idx_te]

    inicio = C.agora()
    pipe = C.montar_pipeline(p["modelo"], p["balanceamento"], p["conjunto"],
                             semente=semente, n_jobs=n_jobs, hiperparametros=hp)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    if p["balanceamento"] == "smotenc":
        sm = pipe.named_steps["smotenc"]
        t_smote, n_apos = sm.tempo_resample_, sm.n_saida_
    else:
        t_smote, n_apos = 0.0, len(ytr)
    clf = pipe.named_steps["classificador"]
    assert list(clf.classes_) == [0, 1], clf.classes_

    t0 = time.perf_counter()
    ypred = pipe.predict(Xte)
    t_pred = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = pipe.predict_proba(Xte)[:, 1]
    t_proba = time.perf_counter() - t0

    # Métricas da Seção 3.6: precisão, sensibilidade (recall) e F1 da classe de
    # ataque; PR-AUC pela precisão média (average precision) e ROC-AUC, a partir
    # das probabilidades; acurácia, TFP e matriz de confusão como complemento.
    # O tempo de treino exclui o SMOTE-NC, registrado à parte (Tabela 7).
    vn, fp, fn, vp = confusion_matrix(yte, ypred, labels=[0, 1]).ravel()
    linha = {
        "id_config": idc, "cenario": p["cenario"], "modelo": p["modelo"],
        "balanceamento": p["balanceamento"], "conjunto_caracteristicas": p["conjunto"],
        "semente": semente,
        "precisao": precision_score(yte, ypred, pos_label=1, zero_division=0),
        "recall": recall_score(yte, ypred, pos_label=1, zero_division=0),
        "f1": f1_score(yte, ypred, pos_label=1, zero_division=0),
        "acuracia": accuracy_score(yte, ypred),
        "taxa_falsos_positivos": fp / (fp + vn) if (fp + vn) else 0.0,
        "roc_auc": roc_auc_score(yte, proba),
        "pr_auc": average_precision_score(yte, proba, pos_label=1),
        "vp": int(vp), "vn": int(vn), "fp": int(fp), "fn": int(fn),
        "tempo_treino_s": t_fit - t_smote, "tempo_predicao_s": t_pred,
        "tempo_smotenc_s": t_smote, "tempo_predict_proba_s": t_proba,
        "tempo_fit_total_s": t_fit,
        "n_treino": len(ytr), "n_treino_ataque": int(ytr.sum()),
        "n_treino_apos_smotenc": int(n_apos),
        "n_teste": len(yte), "n_teste_ataque": int(yte.sum()),
        "hiperparametros": json.dumps(hp), "n_jobs": n_jobs,
        "hostname": os.uname().nodename, "inicio": inicio, "fim": C.agora(),
    }

    # Curvas ROC e PR (pontos), gravadas antes da linha do CSV
    nome = f"{idc}__s{semente}"
    pc = os.path.join(pasta, "curvas")
    os.makedirs(pc, exist_ok=True)
    fpr, tpr, lim = roc_curve(yte, proba, pos_label=1)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "limiar": lim}).to_csv(
        os.path.join(pc, f"{nome}_roc.csv.gz"), index=False)
    pr, rc, lim = precision_recall_curve(yte, proba, pos_label=1)
    pd.DataFrame({"precisao": pr, "recall": rc, "limiar": np.append(lim, np.nan)}).to_csv(
        os.path.join(pc, f"{nome}_pr.csv.gz"), index=False)

    # Importância das características por redução de impureza (MDI) — RF e DT.
    # Comparada com a importância por permutação na Seção 4.7 (Figuras 5 e 6).
    pi = os.path.join(pasta, "importancias")
    os.makedirs(pi, exist_ok=True)
    nomes = list(pipe.named_steps["codificacao"].get_feature_names_out())
    pd.DataFrame({"id_config": idc, "semente": semente, "caracteristica": nomes,
                  "importancia": clf.feature_importances_}).to_csv(
        os.path.join(pi, f"{nome}.csv"), index=False)

    anexar_linha(os.path.join(pasta, "resultados_execucoes.csv"), linha)
    log.info("[%s] semente %d: P=%.4f R=%.4f F1=%.4f PR-AUC=%.4f ROC-AUC=%.4f | "
             "VP=%d FP=%d FN=%d | SMOTE-NC %.1fs, treino %.1fs, predição %.2fs",
             idc, semente, linha["precisao"], linha["recall"], linha["f1"], linha["pr_auc"],
             linha["roc_auc"], vp, fp, fn, t_smote, linha["tempo_treino_s"], t_pred)
    del pipe, clf
    gc.collect()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True, help="CICBellEXFDNS2021.zip ou pasta extraída")
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--n-jobs", type=int, default=6,
                    help="núcleos usados no SMOTE-NC e no Random Forest das execuções "
                         "cronometradas (o mesmo valor em todas as execuções; padrão 6)")
    ap.add_argument("--n-jobs-grid", type=int, default=None,
                    help="processos paralelos do Grid Search (padrão: igual a --n-jobs). "
                         "Reduza se faltar memória; não altera os tempos medidos nas "
                         "execuções principais")
    ap.add_argument("--piloto", action="store_true",
                    help="roda 1 configuração com 1 semente (resultados em <saida>/piloto)")
    ap.add_argument("--config", action="append",
                    help="restringe às configurações indicadas (pode repetir). "
                         "Ex.: heavy__RF__smotenc__stateless14")
    ap.add_argument("--sem-sld", action="store_true",
                    help="experimento extra: inclui Random Forest sem a coluna sld")
    ap.add_argument("--sementes", type=int, default=len(C.SEMENTES),
                    help="número de repetições (padrão 5: sementes 0..4)")
    ap.add_argument("--apenas-grid", action="store_true",
                    help="só faz o ajuste de hiperparâmetros")
    ap.add_argument("--teste-rapido", action="store_true",
                    help="4%% dos dados, grade reduzida, 2 sementes; saída em <saida>_teste")
    ap.add_argument("--forcar-ambiente", action="store_true",
                    help="aceita n_jobs diferente do registrado (não recomendado)")
    a = ap.parse_args()
    if a.n_jobs_grid is None:
        a.n_jobs_grid = a.n_jobs

    base = a.saida + "_teste" if a.teste_rapido else a.saida
    pasta = os.path.join(base, "piloto") if a.piloto else base
    os.makedirs(pasta, exist_ok=True)
    log = C.configurar_log(os.path.join(base, "logs", "experimento.log"))
    log.info("=" * 78)
    log.info("Início: %s", " ".join(sys.argv))
    warnings.filterwarnings("ignore", category=FutureWarning)
    verificar_ambiente(base, a.n_jobs, a.forcar_ambiente, log)

    configs = lista_configs(a.sem_sld)
    if a.config:
        desconhecidas = [c for c in a.config if c not in lista_configs(True)]
        if desconhecidas:
            log.error("Configuração desconhecida: %s. Válidas: %s",
                      desconhecidas, lista_configs(True))
            raise SystemExit(2)
        configs = list(a.config)
    elif a.piloto:
        configs = [C.id_config("heavy", "RF", "smotenc", C.CONJUNTO_PRINCIPAL)]
    sementes = list(range(a.sementes))
    if a.piloto:
        sementes = sementes[:1]
    if a.teste_rapido:
        sementes = sementes[:2]
    log.info("Configurações (%d): %s", len(configs), configs)
    log.info("Sementes: %s | saída: %s", sementes, os.path.abspath(pasta))

    arquivos = C.carregar_arquivos(a.dados)
    dados = {}
    pasta_hp = os.path.join(base, "hiperparametros")
    caminho_csv = os.path.join(pasta, "resultados_execucoes.csv")
    falhas = []
    t_ini = time.perf_counter()
    for idc in configs:
        cen = C.separar_id(idc)["cenario"]
        feitos = ler_feitos(caminho_csv)
        pendentes = [s for s in sementes if (idc, s) not in feitos]
        hp_existe = os.path.exists(os.path.join(pasta_hp, f"{idc}.json"))
        if not pendentes and hp_existe:
            log.info("[%s] completo — pulando.", idc)
            continue
        if cen not in dados:
            dados[cen] = preparar_cenario(arquivos, cen, base, a.teste_rapido, log)
        df, mascara = dados[cen]
        try:
            hp = obter_hiperparametros(idc, df, mascara, pasta_hp, a.n_jobs_grid,
                                       a.teste_rapido, log)
        except Exception:
            log.error("[%s] FALHA no Grid Search:\n%s", idc, traceback.format_exc())
            falhas.append((idc, "grid"))
            continue
        if a.apenas_grid:
            continue
        for s in pendentes:
            try:
                t0 = time.perf_counter()
                executar(idc, hp, df, mascara, s, a.n_jobs, pasta, log)
                log.info("[%s] semente %d concluída em %.1f s (decorrido total %.1f min)",
                         idc, s, time.perf_counter() - t0, (time.perf_counter() - t_ini) / 60)
            except Exception:
                log.error("[%s] FALHA na semente %d:\n%s", idc, s, traceback.format_exc())
                falhas.append((idc, s))
    log.info("Fim. Tempo total desta execução: %.1f min", (time.perf_counter() - t_ini) / 60)
    if falhas:
        log.error("Execuções com falha (rode de novo para tentar só estas): %s", falhas)
        raise SystemExit(1)
    log.info("Nenhuma falha.")


if __name__ == "__main__":
    main()
