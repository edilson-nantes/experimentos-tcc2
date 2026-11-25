# Detecção de exfiltração de dados no tráfego DNS com Machine Learning

Código dos experimentos do Trabalho de Conclusão de Curso:

> NANTES JUNIOR, Edilson Pereira. **Detecção de exfiltração de dados no tráfego DNS com
> Machine Learning: uma avaliação das limitações do conjunto CIC-Bell-DNS-EXF-2021.**
> Trabalho de Conclusão de Curso (Bacharelado em Engenharia de Software) — Universidade
> Tecnológica Federal do Paraná, Cornélio Procópio, 2026. Orientador: Prof. Dr. Lucas Dias
> Hiera Sampaio.

O trabalho compara a **Árvore de Decisão** e o **Random Forest** na detecção de exfiltração
de dados via DNS. Os experimentos usam as **14 características stateless** do
CIC-Bell-DNS-EXF-2021, com os cenários **light** e **heavy** tratados separadamente e com e
sem balanceamento por **SMOTE-NC**: 8 configurações, cada uma repetida 5 vezes. Os
experimentos complementares (Seções 4.4 a 4.9) explicam o resultado principal: quase
todos os vetores de características de ataque também aparecem como benignos, o que
impõe um teto ao desempenho de qualquer modelo.

A versão marcada com a tag `v1.0` corresponde ao texto entregue.

## Estrutura

| Arquivo | O que faz | No TCC |
|---|---|---|
| `comum.py` | leitura dos dados, constantes, **sementes**, pipeline (codificação ordinal → SMOTE-NC → classificador) | Cap. 3 |
| `01_verificar_dados.py` | contagens, tipos, ausentes, duplicatas e vetores nas duas classes | Seção 4.2, Tabela 4 |
| `02_particao_desenvolvimento.py` | separa os 15% de desenvolvimento (estratificado, semente 42) | Seção 3.3 |
| `03_experimento.py` | Grid Search + 5 repetições das 8 configurações (e o RF sem `sld`) | Seções 3.3 a 3.6, 4.1, 4.3, 4.8 |
| `04_analise.py` | média ± desvio-padrão, tabelas e figuras de trabalho do experimento principal | Seção 4.3 |
| `05_teste_equivalencia_smotenc.py` | prova que o SMOTE-NC reescrito no `comum.py` gera as mesmas amostras do imbalanced-learn | Seção 3.3 |
| `comum_complementares.py` | tabela de consulta, divisão por grupos, ajuste do limiar | Seções 4.4 a 4.9 |
| `06_complementares.py` | experimentos complementares | Seções 4.4 a 4.9 |
| `07_analise_complementares.py` | tabelas e figuras de trabalho dos complementares | Seções 4.4 a 4.9 |
| `08_tabelas_figuras_tcc.py` | **Tabelas 3 a 11 e Figuras 3 a 6 exatamente como no TCC** | Cap. 4 |
| `reproduzir_tcc.sh` | roda tudo, na ordem | — |
| `executar_tudo.sh`, `executar_complementares.sh` | rodam 03+04 e 06+07 com até 3 tentativas | — |
| `resultados/` | todos os resultados usados no texto (ver abaixo) | — |
| `docs/` | detalhes de execução e decisões de implementação | — |

## Dados

O conjunto **não é redistribuído** neste repositório. Baixe todos os arquivos do CIC-Bell-DNS-EXF-2021 na
página do Canadian Institute for Cybersecurity
(<https://www.unb.ca/cic/datasets/dns-exf-2021.html>) e coloque todo o conteúdo em um arquivo compactado, sem
extrair, em `dados/`:

```
dados/CICBellEXFDNS2021.zip
```

Os scripts leem o `.zip` diretamente (inclusive os zips internos) e usam só os 18 CSVs
stateless. A pasta extraída também é aceita em `--dados`. Para conferir se os dados são os
mesmos usados no trabalho, compare a impressão digital SHA-256 (item 8 de
`resultados/verificacao/relatorio_verificacao.txt`) com a gravada em
`resultados/particoes/dev_<cenario>.json`; os scripts 03 e 06 fazem essa conferência e
param se ela falhar.

## Instalação

Python **3.13.5** em Linux (os experimentos rodaram no Debian 13). As versões exatas das
bibliotecas estão em `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como reproduzir

### Só as tabelas e figuras, a partir dos resultados já gravados (segundos)

Os resultados de todas as execuções estão em `resultados/`. Para regenerar as tabelas e
figuras do Capítulo 4 sem rodar nenhum modelo:

```bash
python 08_tabelas_figuras_tcc.py --saida resultados
```

A saída vai para `resultados/tcc/`: um `.tex` por tabela, pronto para `\input`, e as figuras
em PDF e PNG com os mesmos nomes dos `\includegraphics` do capítulo.

### Todo o experimento, desde os dados (algumas horas)

```bash
./reproduzir_tcc.sh
```

O script roda 01 → 02 → 03 (com o RF sem `sld`) → 04 → 06 → 07 → 08. É retomável: se
for interrompido, rode de novo e o que já terminou é pulado. Para recalcular do zero, use
uma pasta vazia (por exemplo, renomeie `resultados/`).

**Atenção aos tempos:** os tempos da Tabela 7 dependem do computador. Os números de
detecção (F1, PR-AUC etc.) devem sair idênticos com as mesmas versões das bibliotecas; os
tempos, não. Todas as execuções devem usar o mesmo `n_jobs` (o script recusa outro valor
depois que o primeiro fica registrado em `resultados/ambiente.json`).

Para conferir o código rapidamente (4% dos dados, cerca de 1 minuto, saída em
`resultados_teste/`):

```bash
python 03_experimento.py --dados dados/CICBellEXFDNS2021.zip --teste-rapido --n-jobs 6
python 04_analise.py --saida resultados_teste
```

Instruções detalhadas (piloto, memória, como evitar suspensão do computador em execuções
longas) estão em `docs/experimento_principal.md` e `docs/experimentos_complementares.md`.

## Onde está cada tabela e figura do Capítulo 4

Todas são geradas pelo `08_tabelas_figuras_tcc.py` em `resultados/tcc/`.

| No TCC | Conteúdo | Arquivo gerado | Dados de origem (script) |
|---|---|---|---|
| Tabela 3 (Seção 4.1) | hiperparâmetros do Grid Search | `tabela_03_hiperparametros.tex` | `hiperparametros/*.json` (03) |
| Tabela 4 (Seção 4.2) | vetores de características por classe | `tabela_04_vetores_por_classe.tex` | `verificacao/resumo_vetores.csv` (01) |
| Tabela 5 (Seção 4.3) | métricas no conjunto de teste | `tabela_05_metricas.tex` | `resultados_execucoes.csv` (03) |
| Tabela 6 (Seção 4.3) | matriz de confusão média | `tabela_06_matriz_confusao.tex` | `resultados_execucoes.csv` (03) |
| Tabela 7 (Seção 4.3) | custo computacional | `tabela_07_custo.tex` | `resultados_execucoes.csv` (03) |
| Tabela 8 (Seção 4.4) | linha de referência (tabela de consulta) | `tabela_08_referencia.tex` | `complementares/referencia/` (06) |
| Tabela 9 (Seção 4.5) | divisão aleatória × validação por grupos | `tabela_09_divisao_grupo.tex` | `complementares/grupo/` (06) |
| Tabela 10 (Seção 4.6) | F1 por balanceamento e limiar | `tabela_10_balanceamento.tex` | `complementares/balanceamento/` (06) |
| Tabela 11 (Seção 4.8) | Random Forest com e sem `sld` | `tabela_11_sem_sld.tex` | `resultados_execucoes.csv` (03 `--sem-sld`) |
| Figuras 3 e 4 (Seção 4.4) | curvas ROC e Precision-Recall | `figuras/referencia_light`, `figuras/referencia_heavy` | `curvas/` (03) e `complementares/referencia/` (06) |
| Figuras 5 e 6 (Seção 4.7) | importância das características | `figuras/importancia_light_RF`, `figuras/importancia_light_DT` | `importancias/` (03) e `complementares/permutacao/` (06) |

Os números citados na Seção 4.9 (generalização entre cenários) estão em
`resultados/complementares/analise/tabela_cruzado.tex` (script 07). As Tabelas 1 e 2 são
do Capítulo 3 e não vêm dos experimentos.

## Sementes

Todas as sementes estão no bloco "Sementes" do `comum.py`.

| Semente | Onde é usada |
|---|---|
| `42` (`SEMENTE_PARTICAO_DEV`) | partição de desenvolvimento de 15% (Seção 3.3) |
| `42` (`SEMENTE_GRID`) | dobras da validação cruzada do Grid Search e `random_state` dos modelos no ajuste (Seção 3.4) |
| `0, 1, 2, 3, 4` (`SEMENTES`) | as 5 repetições. Em cada uma, a mesma semente controla a divisão 70/30, o SMOTE-NC e o modelo (Seções 3.3 e 3.5). Os experimentos complementares usam as mesmas sementes e as mesmas divisões |
| `0` (`SEMENTE_TESTE_RAPIDO`) | subamostra do `--teste-rapido` (não entra no TCC) |
| `123` (`SEMENTE_SUBAMOSTRA_EQUIVALENCIA`) | subamostra do teste de equivalência do SMOTE-NC |

Os complementares refazem modelos do experimento principal e conferem se eles reproduzem
exatamente a mesma matriz de confusão e as mesmas métricas
(`resultados/complementares/*/conferencia_principal.csv`). Nas execuções do TCC, não houve
nenhuma divergência.

## O que está em `resultados/`

```
resultados/
  verificacao/                 relatório e tabulações (01); resumo_vetores.csv (Tabela 4)
  particoes/dev_<cenario>.*    índices da partição de desenvolvimento + hash dos dados (02)
  hiperparametros/             escolhidos no Grid Search (+ cv_results/ completo) (03)
  resultados_execucoes.csv     uma linha por (configuração, semente) (03)
  curvas/, importancias/       pontos ROC/PR e importância por impureza de cada execução (03)
  analise/                     tabelas e figuras de trabalho (04)
  complementares/              experimentos das Seções 4.4 a 4.9 (06) e análise (07)
  tcc/                         tabelas e figuras do Capítulo 4 (08)
  ambiente.json                versões, CPU, RAM e n_jobs usados
  logs/                        logs completos das execuções
  piloto/                      execução-piloto (1 configuração, 1 semente)
```
