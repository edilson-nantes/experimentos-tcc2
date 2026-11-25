"""Funções dos experimentos complementares (scripts 06 e 07; Seções 4.4 a 4.9 do TCC 2).

Esses experimentos não estavam no plano do TCC 1. Foram feitos depois do
experimento principal para explicar os resultados dele, a partir da
verificação da Seção 4.2 (vetores idênticos nas duas classes).

Não altera nada do experimento principal: comum.py, 03_experimento.py e
04_analise.py são apenas importados. Como os nomes desses scripts começam com
dígito, são carregados com importlib.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

import comum as C

EXP = importlib.import_module("03_experimento")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
EXPERIMENTOS = ["referencia", "grupo", "sem_duplicatas", "permutacao", "balanceamento",
                "cruzado"]
PASTA_COMPLEMENTARES = "complementares"

MODELO_REF = "REF"                       # tabela de consulta (linha de referência, Seção 4.4)
VARIANTES_REF = ["limiar05", "limiarf1"]

# Variantes da Seção 4.6 (Tabela 10): nome -> (balanceamento usado no pipeline,
# class_weight, ajusta limiar)
# Os hiperparâmetros são os da configuração principal indicada em `balanceamento`.
VARIANTES_BAL = {
    "classweight": ("original", True, False),
    "original_limiar": ("original", False, True),
    "smotenc_limiar": ("smotenc", False, True),
    "classweight_limiar": ("original", True, True),
}

# Arquivos benignos que entram nos dois cenários (excluídos do teste na Seção 4.9)
ARQUIVOS_COMPARTILHADOS = sorted(n for n, (cens, _) in C.ARQUIVOS_ESPERADOS.items()
                                 if len(cens) > 1)

N_DOBRAS_GRUPO = int(round(1 / C.FRACAO_TESTE))  # 3 dobras (cada uma ~1/3 das linhas), Seção 4.5
N_REPETICOES_PERMUTACAO = 10                     # embaralhamentos por semente, Seção 4.7

COLUNAS_EXTRAS = [
    "experimento", "divisao", "dobra", "cenario_teste", "limiar", "f1_oof_limiar",
    "tempo_ajuste_limiar_s", "n_vetores_treino", "n_vetores_teste",
    "n_linhas_teste_vetor_inedito", "n_ataque_teste_vetor_inedito",
]
COLUNAS_CSV = EXP.COLUNAS_CSV + COLUNAS_EXTRAS


# ---------------------------------------------------------------------------
# Vetores das 14 características
# ---------------------------------------------------------------------------
def chave_vetor(df: pd.DataFrame, colunas=None) -> np.ndarray:
    """Identificador (uint64) de cada linha pelo vetor das características.

    Linhas com as mesmas 14 características recebem a mesma chave. Por padrão
    usa sempre as 14 características, inclusive quando o modelo não usa o sld.
    """
    colunas = list(C.CARACTERISTICAS) if colunas is None else list(colunas)
    return pd.util.hash_pandas_object(df[colunas], index=False).to_numpy()


def info_vetores(chave_tr, chave_te, yte) -> dict:
    inedito = ~np.isin(chave_te, np.unique(chave_tr))
    return {
        "n_vetores_treino": int(len(np.unique(chave_tr))),
        "n_vetores_teste": int(len(np.unique(chave_te))),
        "n_linhas_teste_vetor_inedito": int(inedito.sum()),
        "n_ataque_teste_vetor_inedito": int(yte[inedito].sum()),
    }


# ---------------------------------------------------------------------------
# Divisões treino/teste
# ---------------------------------------------------------------------------
def divisao_aleatoria(y: np.ndarray, semente: int):
    """Exatamente a mesma chamada do executar() do 03_experimento.py."""
    return train_test_split(np.arange(len(y)), test_size=C.FRACAO_TESTE,
                            stratify=y, random_state=semente)


def dobras_por_grupo(y: np.ndarray, grupos: np.ndarray, semente: int):
    """Validação cruzada por grupo: cada vetor das 14 características fica
    inteiro numa só dobra. StratifiedGroupKFold com 3 dobras, que tenta manter a
    proporção de classes. Retorna a lista [(idx_treino, idx_teste)] das 3 dobras.

    Usam-se as 3 dobras (predição fora da dobra da base inteira), e não só uma:
    nos dados completos, uma das dobras recebe um único vetor de ataque (o mais
    frequente), o que tornaria o teste degenerado.
    """
    cv = StratifiedGroupKFold(n_splits=N_DOBRAS_GRUPO, shuffle=True, random_state=semente)
    dobras = list(cv.split(np.zeros(len(y)), y, grupos))
    for idx_tr, idx_te in dobras:
        if np.isin(grupos[idx_te], grupos[idx_tr]).any():
            raise RuntimeError("Divisão por grupo com vetores no treino e no teste.")
    return dobras


def base_sem_duplicatas(base: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas repetidas (14 características + classe), mantendo a primeira.

    Um mesmo vetor pode continuar aparecendo duas vezes, uma em cada classe.
    """
    return base.drop_duplicates(subset=list(C.CARACTERISTICAS) + ["classe"],
                                keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabela de consulta (linha de referência do teto)
# ---------------------------------------------------------------------------
class TabelaConsulta:
    """Linha de referência da Seção 4.4 (Tabela 8, Figuras 3 e 4).

    Escore de cada linha = proporção de ataque do seu vetor no treino.
    Vetores que não aparecem no treino recebem a proporção geral de ataque do
    treino (a priori). Não aprende nenhuma regra além da própria tabela de
    vetores; se os modelos empatam com ela, eles só memorizam essa proporção.
    """

    def fit(self, chaves: np.ndarray, y: np.ndarray):
        self.proporcao_ = pd.Series(y.astype(float)).groupby(chaves).mean()
        self.priori_ = float(np.mean(y))
        self.n_vetores_ = int(len(self.proporcao_))
        return self

    def escore(self, chaves: np.ndarray) -> np.ndarray:
        return (pd.Series(chaves).map(self.proporcao_).fillna(self.priori_)
                .to_numpy(dtype=float))


def melhor_limiar_f1(y: np.ndarray, escore: np.ndarray) -> tuple[float, float]:
    """Limiar t que maximiza o F1 da regra `escore >= t`. Empate: o menor t."""
    p, r, t = precision_recall_curve(y, escore, pos_label=1)
    p, r = p[:-1], r[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


# ---------------------------------------------------------------------------
# Pipeline e avaliação
# ---------------------------------------------------------------------------
def construtor_pipeline(modelo, balanceamento, conjunto, semente, n_jobs, hp,
                        class_weight=False):
    """Função sem argumentos que devolve um pipeline novo (comum.montar_pipeline).

    Com class_weight=True, só acrescenta class_weight='balanced' ao classificador.
    """
    def novo():
        p = C.montar_pipeline(modelo, balanceamento, conjunto, semente=semente,
                              n_jobs=n_jobs, hiperparametros=hp)
        if class_weight:
            p.set_params(classificador__class_weight="balanced")
        return p
    return novo


def limiar_por_cv(novo_pipeline, Xtr, ytr, semente) -> tuple[float, float]:
    """Escolhe o limiar de melhor F1 com validação cruzada SÓ no treino (Seção 4.6).

    StratifiedKFold (k=5, embaralhado com a semente da execução): cada dobra é
    prevista por um pipeline ajustado nas outras quatro. O limiar é escolhido
    sobre as probabilidades fora da dobra de todas as linhas do treino.
    """
    oof = np.empty(len(ytr), dtype=float)
    cv = StratifiedKFold(n_splits=C.K_FOLDS, shuffle=True, random_state=semente)
    for a, b in cv.split(np.zeros(len(ytr)), ytr):
        p = novo_pipeline()
        p.fit(Xtr.iloc[a], ytr[a])
        oof[b] = p.predict_proba(Xtr.iloc[b])[:, 1]
        del p
    return melhor_limiar_f1(ytr, oof)


def metricas(yte, ypred, escore) -> dict:
    """Mesmas métricas e mesmas chamadas do executar() do 03_experimento.py."""
    vn, fp, fn, vp = confusion_matrix(yte, ypred, labels=[0, 1]).ravel()
    duas = len(np.unique(yte)) == 2
    return {
        "precisao": precision_score(yte, ypred, pos_label=1, zero_division=0),
        "recall": recall_score(yte, ypred, pos_label=1, zero_division=0),
        "f1": f1_score(yte, ypred, pos_label=1, zero_division=0),
        "acuracia": accuracy_score(yte, ypred),
        "taxa_falsos_positivos": fp / (fp + vn) if (fp + vn) else 0.0,
        "roc_auc": roc_auc_score(yte, escore) if duas else np.nan,
        "pr_auc": average_precision_score(yte, escore, pos_label=1) if duas else np.nan,
        "vp": int(vp), "vn": int(vn), "fp": int(fp), "fn": int(fn),
    }


def ajustar_e_avaliar(novo_pipeline, Xtr, ytr, Xte, yte, semente, ajustar_limiar):
    """Ajusta e avalia como no executar() do 03 (mesmos tempos e métricas).

    Com ajustar_limiar=True, antes do ajuste final escolhe o limiar por CV no
    treino (tempo à parte, em tempo_ajuste_limiar_s), e a predição passa a ser
    predict_proba >= limiar. Retorna (linha, escore, pipeline, predição).
    """
    extra = {"limiar": 0.5, "f1_oof_limiar": np.nan, "tempo_ajuste_limiar_s": 0.0}
    if ajustar_limiar:
        t0 = time.perf_counter()
        lim, f1o = limiar_por_cv(novo_pipeline, Xtr, ytr, semente)
        extra.update(limiar=lim, f1_oof_limiar=f1o,
                     tempo_ajuste_limiar_s=time.perf_counter() - t0)

    pipe = novo_pipeline()
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    if "smotenc" in pipe.named_steps:
        sm = pipe.named_steps["smotenc"]
        t_smote, n_apos = sm.tempo_resample_, sm.n_saida_
    else:
        t_smote, n_apos = 0.0, len(ytr)
    clf = pipe.named_steps["classificador"]
    assert list(clf.classes_) == [0, 1], clf.classes_

    t0 = time.perf_counter()
    if ajustar_limiar:
        ypred = (pipe.predict_proba(Xte)[:, 1] >= extra["limiar"]).astype(int)
    else:
        ypred = pipe.predict(Xte)
    t_pred = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = pipe.predict_proba(Xte)[:, 1]
    t_proba = time.perf_counter() - t0

    linha = metricas(yte, ypred, proba) | extra | {
        "tempo_treino_s": t_fit - t_smote, "tempo_predicao_s": t_pred,
        "tempo_smotenc_s": t_smote, "tempo_predict_proba_s": t_proba,
        "tempo_fit_total_s": t_fit,
        "n_treino": len(ytr), "n_treino_ataque": int(ytr.sum()),
        "n_treino_apos_smotenc": int(n_apos),
        "n_teste": len(yte), "n_teste_ataque": int(yte.sum()),
    }
    return linha, proba, pipe, ypred


def avaliar_referencia(chave_tr, ytr, chave_te, yte):
    """Tabela de consulta. Retorna ([(variante, linha, ypred)], escore).

    limiar05: classe majoritária do vetor (escore > 0,5; empate -> benigno, como
    o predict()). limiarf1: escore >= limiar de melhor F1, escolhido no TREINO.
    """
    t0 = time.perf_counter()
    tab = TabelaConsulta().fit(chave_tr, ytr)
    lim_f1, f1_tr = melhor_limiar_f1(ytr, tab.escore(chave_tr))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    escore = tab.escore(chave_te)
    t_proba = time.perf_counter() - t0
    comum = {
        "tempo_treino_s": t_fit, "tempo_smotenc_s": 0.0, "tempo_fit_total_s": t_fit,
        "tempo_predict_proba_s": t_proba, "tempo_predicao_s": t_proba,
        "n_treino": len(ytr), "n_treino_ataque": int(ytr.sum()),
        "n_treino_apos_smotenc": len(ytr), "n_teste": len(yte),
        "n_teste_ataque": int(yte.sum()), "tempo_ajuste_limiar_s": 0.0,
    }
    saida = []
    for var, ypred, lim, f1o in [("limiar05", (escore > 0.5).astype(int), 0.5, np.nan),
                                 ("limiarf1", (escore >= lim_f1).astype(int), lim_f1, f1_tr)]:
        saida.append((var, metricas(yte, ypred, escore) | comum
                      | {"limiar": lim, "f1_oof_limiar": f1o}, ypred))
    return saida, escore


CAMPOS_MEDIA_DOBRAS = ["tempo_treino_s", "tempo_predicao_s", "tempo_smotenc_s",
                       "tempo_predict_proba_s", "tempo_fit_total_s", "tempo_ajuste_limiar_s",
                       "limiar", "f1_oof_limiar", "n_treino", "n_treino_ataque",
                       "n_treino_apos_smotenc", "n_vetores_treino"]
CAMPOS_SOMA_DOBRAS = ["n_vetores_teste", "n_linhas_teste_vetor_inedito",
                      "n_ataque_teste_vetor_inedito"]


def agregar_dobras(y, ypred, escore, linhas_dobra: list[dict]) -> dict:
    """Linha única a partir da predição fora da dobra (todas as linhas de y).

    Métricas: calculadas sobre a predição/escore fora da dobra. Tempos, limiar e
    tamanhos de treino: média das dobras. Contagens de teste: soma das dobras.
    """
    ln = metricas(y, ypred, escore)
    for k in CAMPOS_MEDIA_DOBRAS:
        v = [float(d[k]) for d in linhas_dobra if k in d and d[k] == d[k]]
        ln[k] = float(np.mean(v)) if v else np.nan
    for k in CAMPOS_SOMA_DOBRAS:
        ln[k] = int(sum(d[k] for d in linhas_dobra))
    ln.update(n_teste=len(y), n_teste_ataque=int(y.sum()), dobra="oof")
    return ln


# ---------------------------------------------------------------------------
# Gravação (mesmo padrão do 03: uma linha por execução, gravada ao terminar)
# ---------------------------------------------------------------------------
def anexar_linha(caminho_csv: str, linha: dict) -> None:
    novo = not os.path.exists(caminho_csv)
    if not novo:
        with open(caminho_csv, encoding="utf-8") as fh:
            cab = next(csv.reader(fh))
        if cab != COLUNAS_CSV:
            raise RuntimeError(f"Cabeçalho de {caminho_csv} difere do esperado.")
    with open(caminho_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUNAS_CSV)
        if novo:
            w.writeheader()
        w.writerow({k: linha.get(k, "") for k in COLUNAS_CSV})
        fh.flush()
        os.fsync(fh.fileno())


def gravar_curvas(pasta: str, nome: str, yte, escore) -> None:
    """Mesmo formato de <saida>/curvas do experimento principal."""
    pc = os.path.join(pasta, "curvas")
    os.makedirs(pc, exist_ok=True)
    if len(np.unique(yte)) < 2:
        return
    fpr, tpr, lim = roc_curve(yte, escore, pos_label=1)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "limiar": lim}).to_csv(
        os.path.join(pc, f"{nome}_roc.csv.gz"), index=False)
    pr, rc, lim = precision_recall_curve(yte, escore, pos_label=1)
    pd.DataFrame({"precisao": pr, "recall": rc, "limiar": np.append(lim, np.nan)}).to_csv(
        os.path.join(pc, f"{nome}_pr.csv.gz"), index=False)


def ler_hiperparametros(base: str, idc: str) -> dict:
    """Lê (sem nunca recalcular) os hiperparâmetros do Grid Search principal."""
    caminho = os.path.join(base, "hiperparametros", f"{idc}.json")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"{caminho} não existe. Os complementares só reaproveitam os hiperparâmetros "
            f"do experimento principal; rode antes o 03_experimento.py.")
    with open(caminho, encoding="utf-8") as fh:
        return json.load(fh)["melhores_hiperparametros"]


# ---------------------------------------------------------------------------
# Conferência contra o experimento principal
# ---------------------------------------------------------------------------
COLUNAS_CONFERENCIA = ["experimento", "id_config", "semente", "conferido", "iguais",
                       "detalhe"]


def conferir(principal: pd.DataFrame | None, idc: str, semente: int, linha: dict,
             campos: list[str], experimento: str, pasta: str, log) -> bool | None:
    """Compara campos de uma execução refeita com a linha do CSV principal.

    Grava o resultado em <pasta>/conferencia_principal.csv e avisa no log se
    algo não bater.
    """
    ok, det = None, "sem linha correspondente no CSV principal"
    if principal is not None:
        r = principal[(principal.id_config == idc) & (principal.semente == semente)]
        if len(r):
            r = r.iloc[-1]
            dif = [f"{c}: principal={r[c]} refeito={linha[c]}" for c in campos
                   if not np.isclose(float(r[c]), float(linha[c]), rtol=0, atol=1e-12)]
            ok = not dif
            det = "; ".join(dif) if dif else "ok"
    if ok is False:
        log.warning("[%s] semente %d NÃO reproduz o experimento principal: %s",
                    idc, semente, det)
    elif ok is None:
        log.warning("[%s] semente %d: %s", idc, semente, det)
    caminho = os.path.join(pasta, "conferencia_principal.csv")
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUNAS_CONFERENCIA)
        if novo:
            w.writeheader()
        w.writerow({"experimento": experimento, "id_config": idc, "semente": semente,
                    "conferido": ok is not None, "iguais": ok, "detalhe": det})
    return ok
