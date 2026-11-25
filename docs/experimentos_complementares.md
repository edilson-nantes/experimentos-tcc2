# Experimentos complementares do TCC 2 (Seções 4.4 a 4.9)

Visão geral e reprodução completa: ver o `README.md` na raiz.

Etapa extra, rodada **depois** do experimento principal. Nenhum arquivo do principal é
alterado: `comum.py`, `01` a `05`, `executar_tudo.sh` e tudo o que já está em
`resultados/` ficam como estão. Os complementares só leem a partição de
desenvolvimento, os hiperparâmetros (`resultados/hiperparametros/*.json`, **nenhum Grid
Search é refeito**) e o CSV principal (usado para conferência). Tudo o que eles gravam
vai para `resultados/complementares/`.

| Arquivo | O que faz |
|---|---|
| `comum_complementares.py` | chave do vetor das 14 características, divisões, tabela de consulta, limiar, métricas |
| `06_complementares.py` | roda os experimentos (retomável, uma linha por execução) |
| `07_analise_complementares.py` | tabelas CSV + LaTeX e figuras, no formato do `04` |
| `executar_complementares.sh` | roda o 06 (até 3 tentativas) e depois o 07 |

Os scripts reaproveitam `comum.py` e as funções do `03` e do `04`, carregadas com
`importlib` porque os nomes começam com dígito: `preparar_cenario`,
`verificar_ambiente`, `ler_feitos`, `resumir`, `fmt`/`ms`, `ler_curvas` e o estilo das
figuras.

## Como rodar

```bash
cd ~/tcc2_experimento
source .venv/bin/activate

# 1. Teste rápido (~2-3 min; usa resultados_teste/, precisa do teste rápido do 03 já feito)
python 06_complementares.py --dados dados/CICBellEXFDNS2021.zip --teste-rapido --n-jobs 6 \
    --experimento referencia --experimento grupo --experimento sem_duplicatas \
    --experimento permutacao --experimento balanceamento --experimento cruzado
python 07_analise_complementares.py --saida resultados_teste

# 2. Completo (estimativa: ~1 h com n_jobs=6)
tmux new -s compl
systemd-inhibit --what=idle:sleep:handle-lid-switch --who=tcc \
    --why="complementares TCC" ./executar_complementares.sh
# opcional: SEM_DUPLICATAS=1 ./executar_complementares.sh
# só alguns: EXPERIMENTOS="referencia grupo" ./executar_complementares.sh
```

Acompanhe com `tail -f resultados/complementares/logs/complementares.log`. Se cair,
rode o mesmo comando: o que já está gravado é pulado. O n_jobs tem de ser o mesmo do
principal (6); o script confere em `resultados/ambiente.json` sem alterá-lo.

**Ordem e custo aproximado:** referência (segundos) → grupo (~15–20 min: 3 ajustes por
semente) → permutação (~15–25 min) → balanceamento (~30–40 min: o ajuste do limiar faz
5 ajustes extras) → cruzado (~5 min).

## O que cada experimento faz

**1. `referencia` — linha de referência do teto.** (Seção 4.4, Tabela 8, Figuras 3 e 4) Mesma divisão 70/30 e mesmas sementes
do principal (a chamada de `train_test_split` é idêntica). No treino, cada vetor das 14
características recebe a proporção de ataque com que aparece; no teste, essa proporção é
o escore. Vetores ausentes do treino recebem a proporção geral do treino. São duas linhas:
`limiar05`, a classe majoritária do vetor (regra de menor erro), e `limiarf1`, o limiar
de maior F1 **escolhido no treino**. Se os modelos empatam com `limiarf1`, eles não fazem
mais do que decorar a proporção de ataque de cada vetor.

**2. `grupo` — otimismo da divisão aleatória.** (Seção 4.5, Tabela 9) As 8 configurações (hiperparâmetros do
principal) e a referência, com `StratifiedGroupKFold` de 3 dobras. O grupo é o vetor das
14 características, de modo que nenhum vetor do teste aparece no treino. **São usadas as
3 dobras** e as métricas são calculadas sobre a predição fora da dobra de toda a base
experimental (uma linha por semente). O motivo: nos dados completos, uma das dobras fica
sempre com **um único vetor de ataque** (o mais frequente, ~13 mil linhas no light e
~78 mil no heavy), o que tornaria degenerado um teste com uma dobra só. As métricas de
cada dobra ficam em `grupo/resultados_por_dobra.csv`. Nesse esquema, a referência não
tem o que consultar (todo vetor do teste é inédito) e vira um escore constante por dobra.

**`sem_duplicatas` (opcional).** Remove as linhas repetidas (14 características + classe)
e faz a divisão aleatória 70/30. Atenção: sobram ~49 (light) e ~90 (heavy) linhas de
ataque na base inteira, e o teste fica com cerca de 15 a 30 linhas de ataque. Por isso,
os resultados são muito instáveis.

**3. `permutacao` — importância por permutação.** (Seção 4.7, Figuras 5 e 6) Refaz os modelos das 8 configurações
(mesma semente → mesmo modelo) e roda `permutation_importance` sobre o **pipeline
inteiro**, permutando as colunas originais (o `sld` é embaralhado como texto, antes do
codificador). Métricas: PR-AUC (principal) e F1; 10 repetições por semente; no conjunto
de teste. O `07` compara com a importância por impureza que o principal já salvou
(`resultados/importancias/`) e calcula a correlação de Spearman entre os ordenamentos.

**4. `balanceamento` — SMOTE-NC × limiar × class_weight.** (Seção 4.6, Tabela 10) Para DT e RF, nos dois
cenários: `classweight` (`class_weight='balanced'`, limiar 0,5), `original_limiar`,
`smotenc_limiar` e `classweight_limiar`. As combinações Original e SMOTE-NC com limiar 0,5
vêm do CSV principal. O limiar é escolhido assim: validação cruzada estratificada de 5
dobras **só no treino**; o limiar é o de maior F1 nas probabilidades fora da dobra; depois,
o pipeline é ajustado no treino inteiro e a predição passa a ser `predict_proba >= limiar`.
O `class_weight` usa os hiperparâmetros da configuração Original. O tempo da validação
cruzada fica em `tempo_ajuste_limiar_s`; o `tempo_treino_s` é o do ajuste final, como no
principal.

**5. `cruzado` — heavy → light e light → heavy.** (Seção 4.9) O modelo é exatamente o do principal
(mesma divisão e mesma semente), testado na base experimental do outro cenário **sem os
arquivos `benign_1` e `benign_2`**, que entram nos dois cenários (sem esse corte, linhas
idênticas estariam no treino e no teste). A proporção de ataque do teste muda, então
compare a PR-AUC sempre com essa proporção (coluna "Ataque no teste").

## Conferência com o principal

Sempre que refaz um modelo do principal (permutação e cruzado) ou usa a mesma divisão
(referência), o 06 compara o resultado com a linha correspondente de
`resultados/resultados_execucoes.csv` (VP, VN, FP, FN, F1, …) e grava
`<experimento>/conferencia_principal.csv`. Qualquer diferença aparece no log e em
`complementares/conferencias_com_divergencia.txt`. O esperado é **nenhuma diferença**.
Se aparecer, verifique se as versões das bibliotecas mudaram (`ambiente.json`).

## O que fica salvo

```
resultados/complementares/
  <experimento>/resultados_execucoes.csv  colunas do CSV principal + experimento, divisao,
                                          dobra, cenario_teste, limiar, f1_oof_limiar,
                                          tempo_ajuste_limiar_s, n_vetores_*, n_*_vetor_inedito
  <experimento>/curvas/                   pontos ROC e PR (mesmo formato do principal)
  grupo/resultados_por_dobra.csv
  permutacao/importancias_permutacao/     uma tabela por (configuração, semente)
  */conferencia_principal.csv
  ambiente.json, logs/complementares.log
  analise/                                tabela_*.tex, resumo_*.csv, figuras/
```

Identificadores (`id_config`, mesmo padrão `cenario__modelo__variante__conjunto`):
`light__REF__limiarf1__stateless14` (referência), `heavy__RF__classweight_limiar__stateless14`
(balanceamento), `heavy_para_light__RF__smotenc__stateless14` (cruzado).

## Tabelas geradas (`resultados/complementares/analise/`)

| Arquivo | Conteúdo |
|---|---|
| `tabela_referencia.tex` | 4 configurações × tabela de consulta (P, R, F1, PR-AUC, ROC-AUC) |
| `tabela_divisao_grupo.tex` | F1, PR-AUC e ROC-AUC: aleatória × por grupo |
| `tabela_divisao_sem_duplicatas.tex` | idem, aleatória × sem duplicatas (se rodado) |
| `tabela_importancia_<cenario>_<modelo>.tex` | impureza × permutação (Original e SMOTE-NC) |
| `tabela_balanceamento.tex`, `..._custo.tex` | 6 combinações × DT/RF × cenário; tempos |
| `tabela_cruzado.tex` | treino em um cenário, teste no outro; F1 na origem para comparar |
| `figuras/referencia_<cenario>` | ROC e PR médias: modelos e tabela de consulta |
| `figuras/importancia_<cenario>_<modelo>` | impureza e permutação lado a lado |

Todas requerem `\usepackage{graphicx}` e usam `\legend{}` com "Fonte: Autoria própria
(2026)", como as do `04`.
