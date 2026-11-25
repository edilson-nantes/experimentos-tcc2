"""Script 6 — experimentos complementares (Seções 4.4 a 4.9 do TCC 2).

Não alteram o experimento principal. Foram feitos para explicar os resultados
dele, depois que a verificação da Seção 4.2 mostrou que quase todos os vetores
de ataque também aparecem como benignos.

Reaproveita a leitura, a partição de desenvolvimento, as sementes 0..4, o
pipeline e os hiperparâmetros JÁ AJUSTADOS do experimento principal (os JSON de
<saida>/hiperparametros são só lidos; nenhum Grid Search é refeito). Tudo é
gravado em <saida>/complementares/<experimento>/.

Experimentos (--experimento, pode repetir; padrão: todos, exceto sem_duplicatas):
  referencia      [Seção 4.4, Tabela 8, Figuras 3 e 4] tabela de consulta: cada vetor das 14 características recebe a
                  proporção de ataque vista no treino (limiar 0,5 e limiar de
                  melhor F1 escolhido no treino). Mesma divisão 70/30 do principal.
  grupo           [Seção 4.5, Tabela 9] as 8 configurações + a referência com validação cruzada por
                  grupo (StratifiedGroupKFold, 3 dobras; grupo = vetor das 14
                  características): nenhum vetor do teste aparece no treino. As
                  métricas são calculadas sobre a predição fora da dobra da base
                  experimental inteira (uma linha por semente); as métricas de
                  cada dobra ficam em resultados_por_dobra.csv.
  sem_duplicatas  (opcional; não usado no TCC) as mesmas, após remover linhas repetidas
                  (14 características + classe) e com divisão aleatória 70/30.
  permutacao      [Seção 4.7, Figuras 5 e 6] importância por permutação (PR-AUC e F1, 10 repetições) no
                  teste, para as 8 configurações. Os modelos refeitos são
                  conferidos contra o CSV principal.
  balanceamento   [Seção 4.6, Tabela 10] class_weight='balanced' e ajuste do limiar (CV de 5 dobras no
                  treino, maximizando o F1), para DT e RF. Original e SMOTE-NC com
                  limiar 0,5 já estão no CSV principal.
  cruzado         [Seção 4.9, só no texto] modelo do cenário A (o mesmo do principal) testado na base
                  experimental do cenário B, sem os arquivos benignos que os dois
                  cenários compartilham.

Uso:
  python 06_complementares.py --dados dados/CICBellEXFDNS2021.zip --teste-rapido
  python 06_complementares.py --dados dados/CICBellEXFDNS2021.zip            # todos
  python 06_complementares.py --dados dados/CICBellEXFDNS2021.zip --experimento referencia
Retomável: o que já está gravado é pulado.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import comum as C
import comum_complementares as CC

EXP = CC.EXP
PADRAO = ["referencia", "grupo", "permutacao", "balanceamento", "cruzado"]


# ---------------------------------------------------------------------------
class Contexto:
    """Dados e parâmetros compartilhados pelos experimentos."""

    def __init__(self, a, base, pasta_c, log):
        self.a, self.base, self.pasta_c, self.log = a, base, pasta_c, log
        self.n_jobs = a.n_jobs
        self.sementes = list(range(a.sementes))[: 2 if a.teste_rapido else None]
        self._arquivos = None
        self._cen = {}
        caminho = os.path.join(base, "resultados_execucoes.csv")
        self.principal = None
        if os.path.exists(caminho):
            p = pd.read_csv(caminho)
            self.principal = p[~p.duplicated(["id_config", "semente"], keep="last")]
        self.falhas = []

    def cenario(self, cen):
        """(base experimental, chave do vetor) do cenário — mesma base do 03."""
        if cen not in self._cen:
            if self._arquivos is None:
                self._arquivos = C.carregar_arquivos(self.a.dados)
            df, mascara = EXP.preparar_cenario(self._arquivos, cen, self.base,
                                               self.a.teste_rapido, self.log)
            base = df.loc[~mascara].reset_index(drop=True)
            self._cen[cen] = (base, CC.chave_vetor(base))
        return self._cen[cen]

    def pasta(self, exp):
        p = os.path.join(self.pasta_c, exp)
        os.makedirs(p, exist_ok=True)
        return p


def configs_principais(cen):
    return [C.id_config(cen, m, b, C.CONJUNTO_PRINCIPAL)
            for m in C.MODELOS for b in C.BALANCEAMENTOS]


def ids_referencia(cen):
    return [C.id_config(cen, CC.MODELO_REF, v, C.CONJUNTO_PRINCIPAL) for v in CC.VARIANTES_REF]


def completar(linha, idc, semente, exp, divisao, cenario_teste, hp, n_jobs, inicio):
    p = C.separar_id(idc)
    return linha | {
        "id_config": idc, "cenario": p["cenario"], "modelo": p["modelo"],
        "balanceamento": p["balanceamento"], "conjunto_caracteristicas": p["conjunto"],
        "semente": semente, "experimento": exp, "divisao": divisao,
        "cenario_teste": cenario_teste, "hiperparametros": json.dumps(hp),
        "n_jobs": n_jobs, "hostname": os.uname().nodename, "inicio": inicio, "fim": C.agora(),
    }


def log_linha(log, idc, s, ln, extra=""):
    log.info("[%s] semente %d: P=%.4f R=%.4f F1=%.4f PR-AUC=%.4f ROC-AUC=%.4f | VP=%d FP=%d "
             "FN=%d | n_teste=%d (ataque %d)%s", idc, s, ln["precisao"], ln["recall"], ln["f1"],
             ln["pr_auc"], ln["roc_auc"], ln["vp"], ln["fp"], ln["fn"], ln["n_teste"],
             ln["n_teste_ataque"], extra)


def tentar(ctx, rotulo, func, *args):
    try:
        func(*args)
    except Exception:
        ctx.log.error("[%s] FALHA:\n%s", rotulo, traceback.format_exc())
        ctx.falhas.append(rotulo)
    gc.collect()


# ---------------------------------------------------------------------------
# Referência (tabela de consulta) sobre uma divisão
# ---------------------------------------------------------------------------
def rodar_referencia(ctx, cen, s, base, chave, idx_tr, idx_te, exp, divisao, info, pasta):
    y = base.classe.values.astype(int)
    inicio = C.agora()
    saida, escore = CC.avaliar_referencia(chave[idx_tr], y[idx_tr], chave[idx_te], y[idx_te])
    for var, ln, _ in saida:
        idc = C.id_config(cen, CC.MODELO_REF, var, C.CONJUNTO_PRINCIPAL)
        ln = completar(ln | info, idc, s, exp, divisao, cen, {}, ctx.n_jobs, inicio)
        CC.gravar_curvas(pasta, f"{idc}__s{s}", y[idx_te], escore)
        CC.anexar_linha(os.path.join(pasta, "resultados_execucoes.csv"), ln)
        log_linha(ctx.log, idc, s, ln, f" | limiar={ln['limiar']:.4f}")
    return saida


# ---------------------------------------------------------------------------
# 1. Referência (tabela de consulta), divisão aleatória do principal — Seção 4.4
# ---------------------------------------------------------------------------
def exp_referencia(ctx):
    pasta = ctx.pasta("referencia")
    csv_ = os.path.join(pasta, "resultados_execucoes.csv")
    for cen in C.CENARIOS:
        for s in ctx.sementes:
            feitos = EXP.ler_feitos(csv_)
            if all((i, s) in feitos for i in ids_referencia(cen)):
                continue
            base, chave = ctx.cenario(cen)
            y = base.classe.values.astype(int)

            def uma():
                idx_tr, idx_te = CC.divisao_aleatoria(y, s)
                info = CC.info_vetores(chave[idx_tr], chave[idx_te], y[idx_te])
                saida = rodar_referencia(ctx, cen, s, base, chave, idx_tr, idx_te,
                                         "referencia", "aleatoria", info, pasta)
                # a divisão tem de ser a mesma do principal
                CC.conferir(ctx.principal, C.id_config(cen, "DT", "original",
                                                       C.CONJUNTO_PRINCIPAL),
                            s, saida[0][1], ["n_teste", "n_teste_ataque", "n_treino_ataque"],
                            "referencia", pasta, ctx.log)
            tentar(ctx, f"referencia/{cen}/s{s}", uma)


# ---------------------------------------------------------------------------
# 2. Divisão por grupo (Seção 4.5) / sem duplicatas (opcional)
# ---------------------------------------------------------------------------
def exp_divisao(ctx, exp):
    pasta = ctx.pasta(exp)
    csv_ = os.path.join(pasta, "resultados_execucoes.csv")
    for cen in C.CENARIOS:
        for s in ctx.sementes:
            feitos = EXP.ler_feitos(csv_)
            pend = [i for i in configs_principais(cen) if (i, s) not in feitos]
            ref_pend = not all((i, s) in feitos for i in ids_referencia(cen))
            if not pend and not ref_pend:
                continue
            base, chave = ctx.cenario(cen)
            if exp == "grupo":
                y = base.classe.values.astype(int)
                dobras = CC.dobras_por_grupo(y, chave, s)
            else:
                base = CC.base_sem_duplicatas(base)
                chave = CC.chave_vetor(base)
                y = base.classe.values.astype(int)
                dobras = [CC.divisao_aleatoria(y, s)]
            infos = []
            for k, (idx_tr, idx_te) in enumerate(dobras):
                infos.append(CC.info_vetores(chave[idx_tr], chave[idx_te], y[idx_te]))
                ctx.log.info("[%s/%s] semente %d dobra %d: n=%d | treino %d (ataque %d) | "
                             "teste %d (ataque %d, %.1f%%; %d vetores de ataque) | linhas de "
                             "teste com vetor inédito: %d", exp, cen, s, k, len(y),
                             len(idx_tr), y[idx_tr].sum(), len(idx_te), y[idx_te].sum(),
                             100 * y[idx_te].mean(),
                             len(np.unique(chave[idx_te][y[idx_te] == 1])),
                             infos[-1]["n_linhas_teste_vetor_inedito"])
            if ref_pend:
                tentar(ctx, f"{exp}/{cen}/REF/s{s}", rodar_dobras, ctx, None, cen, s, base,
                       chave, dobras, infos, exp, pasta)
            for idc in pend:
                tentar(ctx, f"{exp}/{idc}/s{s}", rodar_dobras, ctx, idc, cen, s, base,
                       chave, dobras, infos, exp, pasta)


def rodar_dobras(ctx, idc, cen, s, base, chave, dobras, infos, exp, pasta):
    """Roda uma configuração (ou a referência, se idc=None) em cada dobra.

    Com uma só dobra (sem_duplicatas), grava a linha da própria divisão. Com
    várias (grupo), grava cada dobra em resultados_por_dobra.csv e a linha da
    predição fora da dobra (dobra = "oof") em resultados_execucoes.csv.
    """
    y = base.classe.values.astype(int)
    if idc is None:
        ids = ids_referencia(cen)
        hp = {}
    else:
        ids = [idc]
        p = C.separar_id(idc)
        cat, num = C.colunas_do_conjunto(p["conjunto"])
        X = base[cat + num]
        hp = CC.ler_hiperparametros(ctx.base, idc)
    inicio = C.agora()
    pred = {i: np.zeros(len(y), dtype=int) for i in ids}
    escore = np.zeros(len(y), dtype=float)
    linhas = {i: [] for i in ids}
    for k, (idx_tr, idx_te) in enumerate(dobras):
        if idc is None:
            saida, esc = CC.avaliar_referencia(chave[idx_tr], y[idx_tr], chave[idx_te],
                                               y[idx_te])
            res = [(i, ln, yp) for i, (_, ln, yp) in zip(ids, saida)]
        else:
            novo = CC.construtor_pipeline(p["modelo"], p["balanceamento"], p["conjunto"], s,
                                          ctx.n_jobs, hp)
            ln, esc, pipe, yp = CC.ajustar_e_avaliar(novo, X.iloc[idx_tr], y[idx_tr],
                                                     X.iloc[idx_te], y[idx_te], s,
                                                     ajustar_limiar=False)
            del pipe
            res = [(idc, ln, yp)]
        escore[idx_te] = esc
        for i, ln, yp in res:
            pred[i][idx_te] = yp
            ln = ln | infos[k] | {"dobra": k}
            linhas[i].append(ln)
            if len(dobras) > 1:
                CC.anexar_linha(os.path.join(pasta, "resultados_por_dobra.csv"),
                                completar(ln, i, s, exp, exp, cen, hp, ctx.n_jobs, inicio))
                log_linha(ctx.log, i, s, ln, f" | dobra {k}")
    for i in ids:
        if len(dobras) > 1:
            ln = CC.agregar_dobras(y, pred[i], escore, linhas[i])
            esc_curva, y_curva = escore, y
        else:
            ln = linhas[i][0] | {"dobra": ""}
            esc_curva, y_curva = escore[dobras[0][1]], y[dobras[0][1]]
        ln = completar(ln, i, s, exp, exp, cen, hp, ctx.n_jobs, inicio)
        CC.gravar_curvas(pasta, f"{i}__s{s}", y_curva, esc_curva)
        CC.anexar_linha(os.path.join(pasta, "resultados_execucoes.csv"), ln)
        log_linha(ctx.log, i, s, ln, " | fora da dobra" if len(dobras) > 1 else "")


# ---------------------------------------------------------------------------
# 3. Importância por permutação — Seção 4.7
# ---------------------------------------------------------------------------
def exp_permutacao(ctx):
    pasta = ctx.pasta("permutacao")
    pasta_imp = os.path.join(pasta, "importancias_permutacao")
    os.makedirs(pasta_imp, exist_ok=True)
    for cen in C.CENARIOS:
        for idc in configs_principais(cen):
            for s in ctx.sementes:
                destino = os.path.join(pasta_imp, f"{idc}__s{s}.csv")
                if os.path.exists(destino):
                    continue
                tentar(ctx, f"permutacao/{idc}/s{s}", permutacao_uma, ctx, cen, idc, s,
                       destino, pasta)


def permutacao_uma(ctx, cen, idc, s, destino, pasta):
    base, _ = ctx.cenario(cen)
    p = C.separar_id(idc)
    cat, num = C.colunas_do_conjunto(p["conjunto"])
    X, y = base[cat + num], base.classe.values.astype(int)
    idx_tr, idx_te = CC.divisao_aleatoria(y, s)
    hp = CC.ler_hiperparametros(ctx.base, idc)
    novo = CC.construtor_pipeline(p["modelo"], p["balanceamento"], p["conjunto"], s,
                                  ctx.n_jobs, hp)
    ln, _, pipe, _ = CC.ajustar_e_avaliar(novo, X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te],
                                       y[idx_te], s, ajustar_limiar=False)
    CC.conferir(ctx.principal, idc, s, ln,
                ["vp", "vn", "fp", "fn", "f1", "pr_auc", "roc_auc"], "permutacao", pasta,
                ctx.log)
    t0 = time.perf_counter()
    # Permuta as colunas ORIGINAIS (o sld/longest_word como texto), antes do
    # codificador; o pipeline inteiro é avaliado. n_jobs=1 aqui, porque o RF já
    # usa n_jobs núcleos na predição.
    r = permutation_importance(pipe, X.iloc[idx_te], y[idx_te],
                               scoring={"pr_auc": "average_precision", "f1": "f1"},
                               n_repeats=CC.N_REPETICOES_PERMUTACAO, random_state=s, n_jobs=1)
    t_perm = time.perf_counter() - t0
    base_esc = {"pr_auc": ln["pr_auc"], "f1": ln["f1"]}
    linhas = []
    for metrica, res in r.items():
        for j, carac in enumerate(cat + num):
            linhas.append({"id_config": idc, "semente": s, "caracteristica": carac,
                           "metrica": metrica, "escore_base": base_esc[metrica],
                           "importancia_media": res.importances_mean[j],
                           "importancia_dp": res.importances_std[j],
                           "n_repeticoes": CC.N_REPETICOES_PERMUTACAO,
                           "tempo_permutacao_s": t_perm})
    tmp = destino + ".tmp"
    pd.DataFrame(linhas).to_csv(tmp, index=False)
    os.replace(tmp, destino)
    top = sorted(zip(r["pr_auc"].importances_mean, cat + num), reverse=True)[:3]
    ctx.log.info("[%s] semente %d: permutação em %.1f s | maiores (PR-AUC): %s", idc, s,
                 t_perm, ", ".join(f"{c}={v:.4f}" for v, c in top))


# ---------------------------------------------------------------------------
# 4. SMOTE-NC x limiar x class_weight — Seção 4.6
# ---------------------------------------------------------------------------
def exp_balanceamento(ctx):
    pasta = ctx.pasta("balanceamento")
    csv_ = os.path.join(pasta, "resultados_execucoes.csv")
    for cen in C.CENARIOS:
        for m in C.MODELOS:
            for var, (bal, cw, lim) in CC.VARIANTES_BAL.items():
                idc = C.id_config(cen, m, var, C.CONJUNTO_PRINCIPAL)
                for s in ctx.sementes:
                    if (idc, s) in EXP.ler_feitos(csv_):
                        continue
                    tentar(ctx, f"balanceamento/{idc}/s{s}", balanceamento_uma, ctx, cen, m,
                           var, bal, cw, lim, idc, s, pasta)


def balanceamento_uma(ctx, cen, m, var, bal, cw, lim, idc, s, pasta):
    base, chave = ctx.cenario(cen)
    cat, num = C.colunas_do_conjunto(C.CONJUNTO_PRINCIPAL)
    X, y = base[cat + num], base.classe.values.astype(int)
    idx_tr, idx_te = CC.divisao_aleatoria(y, s)
    # hiperparâmetros da configuração principal correspondente (sem novo Grid Search)
    hp = CC.ler_hiperparametros(ctx.base, C.id_config(cen, m, bal, C.CONJUNTO_PRINCIPAL))
    inicio = C.agora()
    novo = CC.construtor_pipeline(m, bal, C.CONJUNTO_PRINCIPAL, s, ctx.n_jobs, hp,
                                  class_weight=cw)
    ln, proba, _, _ = CC.ajustar_e_avaliar(novo, X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te],
                                        y[idx_te], s, ajustar_limiar=lim)
    hp_reg = hp | ({"class_weight": "balanced"} if cw else {})
    ln = completar(ln | CC.info_vetores(chave[idx_tr], chave[idx_te], y[idx_te]), idc, s,
                   "balanceamento", "aleatoria", cen, hp_reg, ctx.n_jobs, inicio)
    CC.gravar_curvas(pasta, f"{idc}__s{s}", y[idx_te], proba)
    CC.anexar_linha(os.path.join(pasta, "resultados_execucoes.csv"), ln)
    log_linha(ctx.log, idc, s, ln, f" | limiar={ln['limiar']:.4f} "
              f"(ajuste {ln['tempo_ajuste_limiar_s']:.1f} s)")


# ---------------------------------------------------------------------------
# 5. Cruzado heavy <-> light — Seção 4.9
# ---------------------------------------------------------------------------
def exp_cruzado(ctx):
    pasta = ctx.pasta("cruzado")
    csv_ = os.path.join(pasta, "resultados_execucoes.csv")
    for orig, dest in [("heavy", "light"), ("light", "heavy")]:
        par = f"{orig}_para_{dest}"
        ids = {C.id_config(par, m, b, C.CONJUNTO_PRINCIPAL): (m, b)
               for m in C.MODELOS for b in C.BALANCEAMENTOS}
        ids_ref = [C.id_config(par, CC.MODELO_REF, v, C.CONJUNTO_PRINCIPAL)
                   for v in CC.VARIANTES_REF]
        for s in ctx.sementes:
            feitos = EXP.ler_feitos(csv_)
            pend = [i for i in ids if (i, s) not in feitos]
            ref_pend = not all((i, s) in feitos for i in ids_ref)
            if not pend and not ref_pend:
                continue
            base_o, chave_o = ctx.cenario(orig)
            base_d, chave_d = ctx.cenario(dest)
            manter = ~base_d.arquivo.isin(CC.ARQUIVOS_COMPARTILHADOS).to_numpy()
            yo = base_o.classe.values.astype(int)
            yd = base_d.classe.values.astype(int)[manter]
            idx_tr, idx_te = CC.divisao_aleatoria(yo, s)
            info = CC.info_vetores(chave_o[idx_tr], chave_d[manter], yd)
            ctx.log.info("[cruzado %s] semente %d: teste = %d linhas de %s (ataque %d, "
                         "%.1f%%), sem %s | linhas com vetor inédito: %d", par, s, len(yd),
                         dest, yd.sum(), 100 * yd.mean(), CC.ARQUIVOS_COMPARTILHADOS,
                         info["n_linhas_teste_vetor_inedito"])
            if ref_pend:
                def ref():
                    inicio = C.agora()
                    saida, escore = CC.avaliar_referencia(chave_o[idx_tr], yo[idx_tr],
                                                          chave_d[manter], yd)
                    for var, ln, _ in saida:
                        idc = C.id_config(par, CC.MODELO_REF, var, C.CONJUNTO_PRINCIPAL)
                        ln = completar(ln | info, idc, s, "cruzado", "aleatoria", dest, {},
                                       ctx.n_jobs, inicio)
                        CC.gravar_curvas(pasta, f"{idc}__s{s}", yd, escore)
                        CC.anexar_linha(csv_, ln)
                        log_linha(ctx.log, idc, s, ln)
                tentar(ctx, f"cruzado/{par}/REF/s{s}", ref)
            for idc in pend:
                m, b = ids[idc]
                tentar(ctx, f"cruzado/{idc}/s{s}", cruzado_uma, ctx, orig, dest, m, b, idc,
                       s, base_o, idx_tr, idx_te, base_d[manter], info, pasta)


def cruzado_uma(ctx, orig, dest, m, b, idc, s, base_o, idx_tr, idx_te, teste_d, info, pasta):
    cat, num = C.colunas_do_conjunto(C.CONJUNTO_PRINCIPAL)
    Xo, yo = base_o[cat + num], base_o.classe.values.astype(int)
    Xd, yd = teste_d[cat + num], teste_d.classe.values.astype(int)
    idc_orig = C.id_config(orig, m, b, C.CONJUNTO_PRINCIPAL)
    hp = CC.ler_hiperparametros(ctx.base, idc_orig)
    inicio = C.agora()
    novo = CC.construtor_pipeline(m, b, C.CONJUNTO_PRINCIPAL, s, ctx.n_jobs, hp)
    ln_o, _, pipe, _ = CC.ajustar_e_avaliar(novo, Xo.iloc[idx_tr], yo[idx_tr], Xo.iloc[idx_te],
                                         yo[idx_te], s, ajustar_limiar=False)
    # o modelo tem de ser exatamente o do experimento principal
    CC.conferir(ctx.principal, idc_orig, s, ln_o, ["vp", "vn", "fp", "fn", "f1"], "cruzado",
                pasta, ctx.log)
    t0 = time.perf_counter()
    ypred = pipe.predict(Xd)
    t_pred = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = pipe.predict_proba(Xd)[:, 1]
    t_proba = time.perf_counter() - t0
    ln = CC.metricas(yd, ypred, proba) | info | {
        k: ln_o[k] for k in ["tempo_treino_s", "tempo_smotenc_s", "tempo_fit_total_s",
                             "n_treino", "n_treino_ataque", "n_treino_apos_smotenc"]} | {
        "tempo_predicao_s": t_pred, "tempo_predict_proba_s": t_proba,
        "n_teste": len(yd), "n_teste_ataque": int(yd.sum()), "limiar": 0.5,
        "tempo_ajuste_limiar_s": 0.0}
    ln = completar(ln, idc, s, "cruzado", "aleatoria", dest, hp, ctx.n_jobs, inicio)
    CC.gravar_curvas(pasta, f"{idc}__s{s}", yd, proba)
    CC.anexar_linha(os.path.join(pasta, "resultados_execucoes.csv"), ln)
    log_linha(ctx.log, idc, s, ln)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True, help="CICBellEXFDNS2021.zip ou pasta extraída")
    ap.add_argument("--saida", default="resultados",
                    help="pasta do experimento principal (lida; os complementares vão "
                         "para <saida>/complementares)")
    ap.add_argument("--experimento", action="append", choices=CC.EXPERIMENTOS,
                    help="pode repetir; padrão: " + ", ".join(PADRAO))
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--sementes", type=int, default=len(C.SEMENTES))
    ap.add_argument("--teste-rapido", action="store_true",
                    help="4%% dos dados e 2 sementes, sobre <saida>_teste (precisa do "
                         "teste rápido do 03 já feito)")
    ap.add_argument("--forcar-ambiente", action="store_true")
    a = ap.parse_args()

    base = a.saida + "_teste" if a.teste_rapido else a.saida
    pasta_c = os.path.join(base, CC.PASTA_COMPLEMENTARES)
    os.makedirs(pasta_c, exist_ok=True)
    log = C.configurar_log(os.path.join(pasta_c, "logs", "complementares.log"))
    log.info("=" * 78)
    log.info("Início: %s", " ".join(sys.argv))
    warnings.filterwarnings("ignore", category=FutureWarning)

    # n_jobs igual ao do principal (lido, sem alterar o ambiente.json principal)
    amb = os.path.join(base, "ambiente.json")
    if os.path.exists(amb):
        with open(amb, encoding="utf-8") as fh:
            nj = json.load(fh)["n_jobs"]
        if nj != a.n_jobs and not a.forcar_ambiente:
            log.error("n_jobs=%s difere do experimento principal (%s). Use --n-jobs %s.",
                      a.n_jobs, nj, nj)
            raise SystemExit(2)
    EXP.verificar_ambiente(pasta_c, a.n_jobs, a.forcar_ambiente, log)

    ctx = Contexto(a, base, pasta_c, log)
    if ctx.principal is None:
        log.warning("CSV principal não encontrado em %s: as conferências serão puladas.", base)
    exps = a.experimento or PADRAO
    log.info("Experimentos: %s | sementes: %s | saída: %s", exps, ctx.sementes,
             os.path.abspath(pasta_c))
    t_ini = time.perf_counter()
    for exp in [e for e in CC.EXPERIMENTOS if e in exps]:
        t0 = time.perf_counter()
        log.info("-" * 30 + f" {exp} " + "-" * 30)
        if exp == "referencia":
            exp_referencia(ctx)
        elif exp in ("grupo", "sem_duplicatas"):
            exp_divisao(ctx, exp)
        elif exp == "permutacao":
            exp_permutacao(ctx)
        elif exp == "balanceamento":
            exp_balanceamento(ctx)
        elif exp == "cruzado":
            exp_cruzado(ctx)
        log.info("[%s] concluído em %.1f min", exp, (time.perf_counter() - t0) / 60)
    log.info("Fim. Tempo total: %.1f min", (time.perf_counter() - t_ini) / 60)
    conf = os.path.join(pasta_c, "conferencias_com_divergencia.txt")
    divergencias = []
    for exp in CC.EXPERIMENTOS:
        c = os.path.join(pasta_c, exp, "conferencia_principal.csv")
        if os.path.exists(c):
            d = pd.read_csv(c)
            divergencias += [f"{exp}: {r.id_config} s{r.semente}: {r.detalhe}"
                             for r in d.itertuples() if str(r.iguais) == "False"]
    if divergencias:
        with open(conf, "w", encoding="utf-8") as fh:
            fh.write("\n".join(divergencias) + "\n")
        log.warning("%d execuções refeitas NÃO reproduzem o principal (ver %s).",
                    len(divergencias), conf)
    if ctx.falhas:
        log.error("Falhas (rode de novo para tentar só estas): %s", ctx.falhas)
        raise SystemExit(1)
    log.info("Nenhuma falha.")


if __name__ == "__main__":
    main()
