"""Funções e constantes compartilhadas pelos scripts do experimento do TCC 2.

Detecção de exfiltração de dados via DNS (CIC-Bell-DNS-EXF-2021),
14 características stateless, cenários light e heavy tratados separadamente.
Classe positiva (1) = ataque.

Correspondência com o Capítulo 3 (Metodologia) do TCC 2:
  - Seção 3.2 (Dados Analisados): CARACTERISTICAS, ARQUIVOS_ESPERADOS, TABELA1
    e a leitura dos CSVs (carregar_arquivos, montar_cenario).
  - Seção 3.3 (Pré-Processamento e Balanceamento): troca de ausentes por zero,
    remoção do timestamp, partição de desenvolvimento de 15%, divisão 70/30,
    codificação ordinal das colunas textuais e SMOTE-NC (montar_pipeline).
  - Seção 3.4 (Definição e Treinamento dos Modelos): Árvore de Decisão,
    Random Forest e as grades do Grid Search (GRADE_DT, GRADE_RF, K_FOLDS).
  - Seção 3.5 (Desenho Experimental): as 8 configurações e as 5 sementes.
Todas as sementes do trabalho estão reunidas no bloco "Sementes" abaixo.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes do experimento (Cap. 3 do TCC 2)
# ---------------------------------------------------------------------------
# As 14 características stateless (Seção 3.2). As stateful não são usadas:
# cada amostra stateful é uma janela de pacotes, e não um pacote.
CARACTERISTICAS = [
    "FQDN_count", "subdomain_length", "upper", "lower", "numeric", "entropy",
    "special", "labels", "labels_max", "labels_average", "longest_word",
    "sld", "len", "subdomain",
]
# Colunas categóricas textuais, convertidas por codificação ordinal (Seção 3.3)
COLUNAS_TEXTUAIS_ESPERADAS = ["longest_word", "sld"]
# Coluna eliminada antes do treino, para o modelo não memorizar a coleta (Seção 3.3)
COLUNA_TEMPO = "timestamp"

# Fatores do desenho experimental (Seção 3.5, Tabela 2): 2 cenários x 2
# algoritmos x 2 tratamentos de balanceamento = 8 configurações.
CENARIOS = ["light", "heavy"]
MODELOS = ["DT", "RF"]                    # DT = Árvore de Decisão; RF = Random Forest
BALANCEAMENTOS = ["original", "smotenc"]
CONJUNTO_PRINCIPAL = "stateless14"
# Conjunto sem a coluna sld, usado só no experimento extra da Seção 4.8 (Tabela 11)
CONJUNTO_SEM_SLD = "stateless_sem_sld"

FRACAO_DEV = 0.15     # partição de desenvolvimento, só para o Grid Search (Seção 3.3)
FRACAO_TESTE = 0.30   # divisão 70/30 da base experimental (Seção 3.3)
K_FOLDS = 5           # validação cruzada do Grid Search (Seção 3.4)

# ---------------------------------------------------------------------------
# Sementes (todas fixas, para que qualquer pessoa obtenha os mesmos números)
# ---------------------------------------------------------------------------
# Partição de desenvolvimento de 15%, separada uma única vez (Seção 3.3).
SEMENTE_PARTICAO_DEV = 42
# Dobras da validação cruzada do Grid Search e random_state dos modelos
# durante o ajuste de hiperparâmetros (Seção 3.4).
SEMENTE_GRID = 42
# As 5 repetições do experimento (Seções 3.3 e 3.5). Em cada repetição, a
# mesma semente controla a divisão 70/30, o SMOTE-NC e o modelo; os
# experimentos complementares (Seções 4.4 a 4.9) reutilizam essas sementes.
SEMENTES = [0, 1, 2, 3, 4]
# Subamostra de 4% usada só no --teste-rapido (conferência do código, não
# entra no TCC).
SEMENTE_TESTE_RAPIDO = 0
# Subamostra usada só no 05_teste_equivalencia_smotenc.py.
SEMENTE_SUBAMOSTRA_EQUIVALENCIA = 123

# Espaço de busca do Grid Search (Seção 3.4)
GRADE_RF = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}
GRADE_DT = {
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}

# Tabela 1 do TCC 2 (contagens esperadas de amostras stateless, Seção 3.2).
# O script 01 confere os dados lidos contra estes números (Seção 4.2).
TABELA1 = {
    "light": {"benigno": 281_164, "ataque": 42_683},
    "heavy": {"benigno": 402_767, "ataque": 251_670},
}

# Os 18 arquivos stateless do conjunto. Os dois arquivos benign_1/benign_2
# (Benign.zip na raiz da pasta CSV) entram nos DOIS cenários: só assim as
# contagens fecham com a Tabela 1 (verificado nos arquivos originais).
ARQUIVOS_ESPERADOS = {
    "stateless_features-light_audio.pcap.csv": (("light",), 1),
    "stateless_features-light_compressed.pcap.csv": (("light",), 1),
    "stateless_features-light_exe.pcap.csv": (("light",), 1),
    "stateless_features-light_image.pcap.csv": (("light",), 1),
    "stateless_features-light_text.pcap.csv": (("light",), 1),
    "stateless_features-light_video.pcap.csv": (("light",), 1),
    "stateless_features-light_benign.pcap.csv": (("light",), 0),
    "stateless_features-heavy_audio.pcap.csv": (("heavy",), 1),
    "stateless_features-heavy_compressed.pcap.csv": (("heavy",), 1),
    "stateless_features-heavy_exe.pcap.csv": (("heavy",), 1),
    "stateless_features-heavy_image.pcap.csv": (("heavy",), 1),
    "stateless_features-heavy_text.pcap.csv": (("heavy",), 1),
    "stateless_features-heavy_video.pcap.csv": (("heavy",), 1),
    "stateless_features-benign_heavy_1.pcap.csv": (("heavy",), 0),
    "stateless_features-benign_heavy_2.pcap.csv": (("heavy",), 0),
    "stateless_features-benign_heavy_3.pcap.csv": (("heavy",), 0),
    "stateless_features-benign_1.pcap.csv": (("light", "heavy"), 0),
    "stateless_features-benign_2.pcap.csv": (("light", "heavy"), 0),
}

_PADRAO_STATELESS = re.compile(r"^stateless_features-.+\.csv$")


# ---------------------------------------------------------------------------
# Leitura dos arquivos (aceita o .zip original, a pasta extraída, ou zips internos)
# ---------------------------------------------------------------------------
def _coletar_bytes(caminho: str) -> dict[str, tuple[str, bytes]]:
    """Retorna {nome_do_arquivo: (origem, conteúdo)} para os CSVs stateless.

    Percorre zips aninhados. Ignora a pasta PCAP.
    """
    fontes: dict[str, tuple[str, bytes]] = {}

    def adicionar(nome: str, origem: str, conteudo: bytes) -> None:
        base = os.path.basename(nome)
        if not _PADRAO_STATELESS.match(base):
            return
        if base in fontes:
            if fontes[base][1] != conteudo:
                raise RuntimeError(
                    f"Arquivo {base} encontrado duas vezes com conteúdos diferentes:\n"
                    f"  {fontes[base][0]}\n  {origem}\nDeixe apenas uma cópia dos dados."
                )
            return  # mesma cópia (zip e pasta extraída): ignora
        fontes[base] = (origem, conteudo)

    def varrer_zip(origem_zip, rotulo: str) -> None:
        with zipfile.ZipFile(origem_zip) as z:
            for info in z.infolist():
                nome = info.filename
                if info.is_dir() or "PCAP" in nome.upper().split("/"):
                    continue
                if nome.lower().endswith(".zip"):
                    varrer_zip(io.BytesIO(z.read(info)), f"{rotulo}!{nome}")
                elif nome.lower().endswith(".csv"):
                    adicionar(nome, f"{rotulo}!{nome}", z.read(info))

    if os.path.isfile(caminho) and caminho.lower().endswith(".zip"):
        varrer_zip(caminho, caminho)
    elif os.path.isdir(caminho):
        for raiz, dirs, arquivos in os.walk(caminho):
            dirs[:] = sorted(d for d in dirs if d.upper() != "PCAP")
            for a in sorted(arquivos):
                p = os.path.join(raiz, a)
                if a.lower().endswith(".zip"):
                    varrer_zip(p, p)
                elif a.lower().endswith(".csv"):
                    with open(p, "rb") as fh:
                        adicionar(a, p, fh.read())
    else:
        raise FileNotFoundError(f"Caminho de dados não encontrado: {caminho}")
    return fontes


def ler_csv_stateless(conteudo: bytes) -> pd.DataFrame:
    """Lê um CSV stateless com o tratamento correto de textos.

    IMPORTANTE: keep_default_na=False. Com a leitura padrão do pandas, as
    palavras literais "nan", "null", "NA" etc. da coluna longest_word (ex.: o
    domínio "nanit" tem longest_word = "nan") viram valores ausentes falsos.
    Só a célula vazia é tratada como ausente.
    """
    return pd.read_csv(
        io.BytesIO(conteudo),
        keep_default_na=False,
        na_values=[""],
        dtype={c: str for c in COLUNAS_TEXTUAIS_ESPERADAS},
        low_memory=False,
    )


def carregar_arquivos(caminho_dados: str) -> dict[str, pd.DataFrame]:
    """Lê os 18 arquivos stateless. Retorna {nome: DataFrame bruto}."""
    fontes = _coletar_bytes(caminho_dados)
    faltando = sorted(set(ARQUIVOS_ESPERADOS) - set(fontes))
    extras = sorted(set(fontes) - set(ARQUIVOS_ESPERADOS))
    if faltando:
        raise RuntimeError("Arquivos stateless ausentes: " + ", ".join(faltando))
    if extras:
        logging.getLogger("tcc").warning("Arquivos stateless não esperados (ignorados): %s", extras)
    return {nome: ler_csv_stateless(fontes[nome][1]) for nome in sorted(ARQUIVOS_ESPERADOS)}


def bytes_dos_arquivos(caminho_dados: str) -> dict[str, tuple[str, bytes]]:
    return _coletar_bytes(caminho_dados)


def montar_cenario(arquivos: dict[str, pd.DataFrame], cenario: str) -> pd.DataFrame:
    """Concatena os arquivos de um cenário em ordem determinística.

    Retorna DataFrame com as 14 características + classe + arquivo + linha.
    A coluna timestamp é eliminada e ausentes são substituídos por zero
    (Seção 3.3, seguindo Mahdavifar et al., 2021). Os CSVs não têm coluna
    de rótulo: a classe vem do arquivo (ARQUIVOS_ESPERADOS).
    """
    partes = []
    for nome in sorted(arquivos):
        cens, classe = ARQUIVOS_ESPERADOS[nome]
        if cenario not in cens:
            continue
        d = arquivos[nome]
        faltam = [c for c in CARACTERISTICAS if c not in d.columns]
        if faltam:
            raise RuntimeError(f"{nome}: colunas ausentes {faltam}")
        d = d[CARACTERISTICAS].copy()
        d["classe"] = np.int8(classe)
        d["arquivo"] = nome
        d["linha"] = np.arange(len(d), dtype=np.int64)
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    for c in CARACTERISTICAS:
        if c in COLUNAS_TEXTUAIS_ESPERADAS:
            df[c] = df[c].fillna("0").astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="raise").fillna(0)
    return df


def carregar_cenario(caminho_dados: str, cenario: str) -> pd.DataFrame:
    return montar_cenario(carregar_arquivos(caminho_dados), cenario)


def hash_dados(df: pd.DataFrame) -> str:
    """Impressão digital estável dos dados (características + classe, em ordem)."""
    texto = df[CARACTERISTICAS + ["classe"]].to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline (imblearn) — OrdinalEncoder -> [SMOTE-NC] -> classificador
# ---------------------------------------------------------------------------
# Seção 3.3: o SMOTE-NC fica DENTRO do pipeline. Assim ele só é aplicado à
# parcela de treinamento (no Grid Search, a cada dobra de treino da validação
# cruzada), e o conjunto de teste nunca recebe amostras sintéticas (sem data
# leakage). O tempo do SMOTE-NC é medido à parte (Seção 3.5, Tabela 7).
def _classe_smotenc_cronometrado():
    import math

    import imblearn
    from imblearn.over_sampling import SMOTENC
    from imblearn.over_sampling._smote import base as _b
    from scipy import sparse
    from sklearn.utils import check_random_state

    class SMOTENCCronometrado(SMOTENC):
        """SMOTE-NC que registra o próprio tempo de execução (tempo_resample_).

        Também substitui SMOTENC._generate_samples por uma versão equivalente
        que não densifica o one-hot das colunas textuais. A original faz
        nn_data.toarray() e monta um array (n_novas x k x n_categorias); com as
        ~85 mil categorias de sld/longest_word isso passa de dezenas de GB. Aqui
        a categoria mais frequente entre os k vizinhos é obtida pelos códigos
        das categorias, e o desempate usa o mesmo gerador aleatório, consumido
        da mesma forma (rng.permutation sobre os mesmos pares linha/coluna, na
        mesma ordem) — o resultado é idêntico ao do imblearn (verificado em
        teste comparando as duas implementações: 05_teste_equivalencia_smotenc.py).

        Ou seja, o método é o SMOTE-NC descrito na Seção 3.3 (Chawla et al.,
        2002); a reescrita só reduz memória e tempo para caber num computador
        com 11,6 GB de RAM (Seção 4.1).
        """

        def _fit_resample(self, X, y):
            # Cópia de SMOTENC._fit_resample (imblearn 0.14.2) com UMA alteração,
            # marcada abaixo. Na original, `X_class[:, n_cont:] = X_class_categorical`
            # faz o scipy montar uma grade densa (amostras x categorias) de índices,
            # o que estoura a memória com ~85 mil categorias.
            self.n_features_ = _b._num_features(X)
            self._validate_column_types(X)
            self._validate_estimator()

            X_continuous = _b._safe_indexing(X, self.continuous_features_, axis=1)
            X_continuous = _b.check_array(X_continuous, accept_sparse=["csr", "csc"])
            X_categorical = _b._safe_indexing(X, self.categorical_features_, axis=1)
            if X_continuous.dtype.name != "object":
                dtype_ohe = X_continuous.dtype
            else:
                dtype_ohe = np.float64

            if self.categorical_encoder is None:
                self.categorical_encoder_ = _b.OneHotEncoder(
                    handle_unknown="ignore", dtype=dtype_ohe)
            else:
                self.categorical_encoder_ = _b.clone(self.categorical_encoder)

            X_ohe = self.categorical_encoder_.fit_transform(
                X_categorical.toarray() if sparse.issparse(X_categorical) else X_categorical)
            if not sparse.issparse(X_ohe):
                X_ohe = sparse.csr_matrix(X_ohe, dtype=dtype_ohe)
            n_categories = sum(c.size for c in self.categorical_encoder_.categories_)
            if X_ohe.shape[1] != n_categories:
                raise ValueError("SMOTENC requires a one-hot encoding with one column per "
                                 "category for the categorical features.")

            X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr", dtype=dtype_ohe)
            X_resampled = [X_encoded.copy()]
            y_resampled = [y.copy()]

            self.median_std_ = {}
            for class_sample, n_samples in self.sampling_strategy_.items():
                if n_samples == 0:
                    continue
                target_class_indices = np.flatnonzero(y == class_sample)
                X_class = _b._safe_indexing(X_encoded, target_class_indices)

                _, var = _b.csr_mean_variance_axis0(
                    X_class[:, : self.continuous_features_.size])
                self.median_std_[class_sample] = np.median(np.sqrt(var))

                if math.isclose(self.median_std_[class_sample], 0):
                    self._X_categorical_minority_encoded = X_class[
                        :, self.continuous_features_.size:].toarray()

                # --- ALTERAÇÃO -----------------------------------------------------
                # Original:
                #   X_class_categorical = X_class[:, n_cont:]
                #   X_class_categorical.data[:] = median_std / sqrt(2)
                #   X_class[:, n_cont:] = X_class_categorical
                # Equivalente: troca diretamente os valores armazenados nas colunas
                # categóricas (índice >= n_cont), mantendo a estrutura CSR.
                X_class = sparse.csr_matrix(X_class, copy=True)
                X_class.data[X_class.indices >= self.continuous_features_.size] = (
                    self.median_std_[class_sample] / np.sqrt(2))
                # -------------------------------------------------------------------

                # --- ALTERAÇÃO 2 ---------------------------------------------------
                # Original: self.nn_k_.fit(X_class); nns = self.nn_k_.kneighbors(...)
                # Busca exaustiva em matriz esparsa com dezenas de milhares de
                # colunas: horas no cenário heavy. Como os ataques têm poucos
                # vetores distintos, os vizinhos são buscados entre os vetores
                # distintos (mesma métrica do SMOTE-NC) — ver _vizinhos_por_grupo.
                nns = self._vizinhos_por_grupo(X_class)
                # -------------------------------------------------------------------
                X_new, y_new = self._make_samples(
                    X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0)
                X_resampled.append(X_new)
                y_resampled.append(y_new)

            X_resampled = sparse.vstack(X_resampled, format=X_encoded.format)
            y_resampled = np.hstack(y_resampled)

            X_res_cat = X_resampled[:, self.continuous_features_.size:]
            X_res_cat.data = np.ones_like(X_res_cat.data)
            X_res_cat_dec = self.categorical_encoder_.inverse_transform(X_res_cat)

            if sparse.issparse(X):
                X_resampled = sparse.hstack(
                    (X_resampled[:, : self.continuous_features_.size], X_res_cat_dec),
                    format="csr")
            else:
                X_resampled = np.hstack(
                    (X_resampled[:, : self.continuous_features_.size].toarray(),
                     X_res_cat_dec))

            indices_reordered = np.argsort(
                np.hstack((self.continuous_features_, self.categorical_features_)))
            if sparse.issparse(X_resampled):
                col_indices = X_resampled.indices.copy()
                for idx, col_idx in enumerate(indices_reordered):
                    mask = X_resampled.indices == col_idx
                    col_indices[mask] = idx
                X_resampled.indices = col_indices
            else:
                X_resampled = X_resampled[:, indices_reordered]

            return X_resampled, y_resampled

        MAX_VETORES_DISTINTOS = 1000

        def _vizinhos_por_grupo(self, X_class):
            """k vizinhos mais próximos (sem o próprio ponto) de cada amostra.

            Equivale a NearestNeighbors(k+1).kneighbors(X_class)[:, 1:] com a
            métrica euclidiana sobre a codificação do SMOTE-NC, mas calcula as
            distâncias só entre os vetores distintos. Amostras idênticas estão a
            distância 0 e vêm primeiro; depois, membros dos vetores distintos
            mais próximos, em ordem de distância. Qual cópia de um mesmo vetor é
            escolhida não altera as amostras geradas (os valores são iguais). A
            única diferença possível em relação à busca exaustiva é o desempate
            entre dois vetores DIFERENTES exatamente à mesma distância.
            Com muitos vetores distintos, usa a busca original.
            """
            k1 = self.nn_k_.n_neighbors          # k + 1 (inclui o próprio ponto)
            Xc = sparse.csr_matrix(X_class)
            Xc.sort_indices()
            # chave de cada linha = seus índices e valores armazenados
            chaves = [(Xc.indices[a:b].tobytes(), Xc.data[a:b].tobytes())
                      for a, b in zip(Xc.indptr[:-1], Xc.indptr[1:])]
            grupo_de = {}
            grupo = np.empty(len(chaves), dtype=np.int64)
            for i, ch in enumerate(chaves):
                grupo[i] = grupo_de.setdefault(ch, len(grupo_de))
            n_grupos = len(grupo_de)
            if n_grupos > self.MAX_VETORES_DISTINTOS:
                self.nn_k_.fit(X_class)
                return self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            if len(grupo) < k1:
                raise ValueError(f"Expected n_neighbors <= n_samples_fit, but "
                                 f"n_neighbors = {k1}, n_samples_fit = {len(grupo)}")
            _, rep = np.unique(grupo, return_index=True)     # 1 representante/grupo
            R = Xc[rep]
            R = R[:, np.unique(R.indices)].toarray()  # só as colunas usadas
            # distâncias euclidianas exatas entre representantes (diferença direta)
            dist = np.empty((n_grupos, n_grupos))
            for g in range(n_grupos):
                dist[g] = np.sqrt(((R - R[g]) ** 2).sum(axis=1))
            ordem_grupos = np.argsort(dist, axis=1, kind="stable")
            membros = np.argsort(grupo, kind="stable")
            ini = np.searchsorted(grupo[membros], np.arange(n_grupos))
            tam = np.bincount(grupo, minlength=n_grupos)
            # para cada grupo: lista de k1 candidatos (o próprio grupo vem primeiro,
            # pois está a distância 0)
            cand = np.empty((n_grupos, k1), dtype=np.int64)
            for g in range(n_grupos):
                lst = []
                for h in ordem_grupos[g]:
                    if h == g:
                        continue
                    if not lst:
                        lst.extend(membros[ini[g]:ini[g] + min(tam[g], k1)])
                    lst.extend(membros[ini[h]:ini[h] + min(tam[h], k1 - len(lst))])
                    if len(lst) >= k1:
                        break
                if not lst:  # um único grupo
                    lst.extend(membros[ini[g]:ini[g] + k1])
                cand[g] = lst[:k1]
            c = cand[grupo]                                   # (n, k1)
            eh_proprio = c == np.arange(len(grupo))[:, None]
            # remove o próprio ponto; se ele não estiver entre os candidatos
            # (grupo com mais de k1 cópias), remove o último
            sem_proprio = ~eh_proprio.any(axis=1)
            eh_proprio[sem_proprio, k1 - 1] = True
            return c[~eh_proprio].reshape(len(grupo), k1 - 1)

        def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_type, y=None):
            n_cont = self.continuous_features_.size
            c = self.median_std_[y_type] / np.sqrt(2)
            # Casos de borda em que a equivalência não é garantida (mediana do
            # desvio padrão nula ou ínfima): usa a implementação original.
            if (math.isclose(self.median_std_[y_type], 0) or c < 1e-6
                    or not sparse.issparse(nn_data)):
                return super()._generate_samples(X, nn_data, nn_num, rows, cols, steps,
                                                 y_type, y)
            rng = check_random_state(self.random_state)
            # Parte contínua: mesma interpolação do SMOTE (classe base do SMOTENC).
            X_new = super(SMOTENC, self)._generate_samples(X, nn_data, nn_num, rows, cols,
                                                           steps)
            X_new = sparse.csr_matrix(X_new)
            vizinhos = nn_num[rows]                      # (n_novas, k)
            n_novas = vizinhos.shape[0]
            nn_csr = sparse.csr_matrix(nn_data)
            tamanhos = [cat.size for cat in self.categorical_encoder_.categories_]
            inicios = n_cont + np.concatenate(([0], np.cumsum(tamanhos)[:-1]))
            blocos_cat = []
            for ini, tam in zip(inicios, tamanhos):
                bloco = nn_csr[:, ini:ini + tam].tocsr()
                bloco.eliminate_zeros()
                bloco.sort_indices()
                if not (np.diff(bloco.indptr) == 1).all():
                    raise RuntimeError("SMOTE-NC: esperada exatamente uma categoria por "
                                       "amostra em cada coluna textual.")
                codigos = bloco.indices                  # categoria de cada amostra
                viz = np.sort(codigos[vizinhos], axis=1)  # (n_novas, k), ordenado
                contagem = (viz[:, :, None] == viz[:, None, :]).sum(axis=2)
                primeira = np.ones_like(viz, dtype=bool)
                primeira[:, 1:] = viz[:, 1:] != viz[:, :-1]
                eh_max = primeira & (contagem == contagem.max(axis=1, keepdims=True))
                # mesmos pares (linha, categoria) de np.argwhere(is_max) no original,
                # na mesma ordem (linhas crescentes, categorias crescentes)
                lin, pos = np.nonzero(eh_max)
                max_idxs = rng.permutation(np.column_stack((lin, viz[lin, pos])))
                xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
                col_sels = max_idxs[idx_sels, 1]
                blocos_cat.append(sparse.csr_matrix(
                    (np.ones(len(xs), dtype=X_new.dtype), (xs, col_sels)),
                    shape=(n_novas, tam)))
            return sparse.hstack([X_new[:, :n_cont]] + blocos_cat, format="csr",
                                 dtype=X_new.dtype)

        def fit_resample(self, X, y, **params):
            t0 = time.perf_counter()
            Xr, yr = super().fit_resample(X, y, **params)
            self.tempo_resample_ = time.perf_counter() - t0
            self.n_entrada_ = int(len(y))
            self.n_saida_ = int(len(yr))
            return Xr, yr

    if imblearn.__version__ != "0.14.2":
        logging.getLogger("tcc").warning(
            "SMOTENCCronometrado foi escrito e verificado para o imblearn 0.14.2 "
            "(instalado: %s). Refaça a comparação com o SMOTENC original.",
            imblearn.__version__)

    SMOTENCCronometrado.__module__ = __name__
    SMOTENCCronometrado.__qualname__ = "SMOTENCCronometrado"
    return SMOTENCCronometrado


try:  # definida no nível do módulo para poder ser serializada (n_jobs > 1)
    SMOTENCCronometrado = _classe_smotenc_cronometrado()
except ImportError:  # imblearn ausente: o script 01 continua funcionando
    SMOTENCCronometrado = None


def colunas_do_conjunto(conjunto: str) -> tuple[list[str], list[str]]:
    """Retorna (colunas_textuais, colunas_numericas) do conjunto de características."""
    if conjunto == CONJUNTO_PRINCIPAL:
        cols = list(CARACTERISTICAS)
    elif conjunto == CONJUNTO_SEM_SLD:
        cols = [c for c in CARACTERISTICAS if c != "sld"]
    else:
        raise ValueError(conjunto)
    cat = [c for c in cols if c in COLUNAS_TEXTUAIS_ESPERADAS]
    num = [c for c in cols if c not in COLUNAS_TEXTUAIS_ESPERADAS]
    return cat, num


def montar_pipeline(modelo: str, balanceamento: str, conjunto: str, semente: int,
                    n_jobs: int, hiperparametros: dict | None = None):
    from imblearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.tree import DecisionTreeClassifier

    """Monta o pipeline de uma configuração (Seções 3.3 e 3.4).

    1. Codificação ordinal (Label Encoding) de sld e longest_word, ajustada só
       no treino; categorias que aparecem apenas no teste recebem o código -1.
       Escolhida em vez do One-Hot por causa da alta cardinalidade (Seção 3.3).
    2. SMOTE-NC (só se balanceamento == "smotenc"), com as colunas textuais
       indicadas como categóricas, até igualar as classes (1:1).
    3. Árvore de Decisão ou Random Forest, com os hiperparâmetros do Grid Search.
    """
    cat, num = colunas_do_conjunto(conjunto)
    # As colunas textuais saem primeiro do ColumnTransformer (índices 0..len(cat)-1),
    # seguidas das numéricas (remainder="passthrough" na ordem de `num`).
    codificador = ColumnTransformer(
        [("ordinal",
          OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          cat)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    passos = [("codificacao", codificador)]
    if balanceamento == "smotenc":
        passos.append(("smotenc", SMOTENCCronometrado(
            categorical_features=list(range(len(cat))),
            sampling_strategy=1.0,  # minoria até igualar a maioria (1:1)
            # k=5 vizinhos (padrão do SMOTE). n_neighbors=6 porque o próprio
            # ponto é devolvido como vizinho mais próximo e descartado.
            k_neighbors=NearestNeighbors(n_neighbors=6, n_jobs=n_jobs),
            random_state=semente,
        )))
    if modelo == "DT":
        clf = DecisionTreeClassifier(random_state=semente)
    elif modelo == "RF":
        clf = RandomForestClassifier(random_state=semente, n_jobs=n_jobs)
    else:
        raise ValueError(modelo)
    passos.append(("classificador", clf))
    pipe = Pipeline(passos)
    if hiperparametros:
        pipe.set_params(**{f"classificador__{k}": v for k, v in hiperparametros.items()})
    return pipe


def id_config(cenario: str, modelo: str, balanceamento: str, conjunto: str) -> str:
    return f"{cenario}__{modelo}__{balanceamento}__{conjunto}"


def separar_id(idc: str) -> dict:
    c, m, b, k = idc.split("__")
    return {"cenario": c, "modelo": m, "balanceamento": b, "conjunto": k}


# ---------------------------------------------------------------------------
# Ambiente, log, utilidades
# ---------------------------------------------------------------------------
def versoes_bibliotecas() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "imblearn", "joblib", "matplotlib"]:
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def info_hardware() -> dict:
    info = {
        "plataforma": platform.platform(),
        "maquina": platform.machine(),
        "hostname": platform.node(),
        "nucleos_logicos": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            txt = fh.read()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        info["cpu"] = m.group(1).strip() if m else None
        pares = set()
        for bloco in txt.strip().split("\n\n"):
            pid = re.search(r"physical id\s*:\s*(\d+)", bloco)
            cid = re.search(r"core id\s*:\s*(\d+)", bloco)
            if pid and cid:
                pares.add((pid.group(1), cid.group(1)))
        info["nucleos_fisicos"] = len(pares) or None
    except OSError:
        info["cpu"] = platform.processor()
    try:
        with open("/proc/meminfo") as fh:
            m = re.search(r"MemTotal:\s*(\d+) kB", fh.read())
        info["ram_gb"] = round(int(m.group(1)) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True,
                                       timeout=10).stdout
    except Exception:
        pass
    return info


def configurar_log(caminho_log: str, nome: str = "tcc") -> logging.Logger:
    os.makedirs(os.path.dirname(caminho_log), exist_ok=True)
    log = logging.getLogger(nome)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(caminho_log, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def gravar_json(caminho: str, obj) -> None:
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, caminho)


def agora() -> str:
    return datetime.now().isoformat(timespec="seconds")
