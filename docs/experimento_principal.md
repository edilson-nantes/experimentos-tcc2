# Experimento do TCC 2: detecção de exfiltração DNS (DT x RF)

Visão geral e reprodução completa: ver o `README.md` na raiz. Este arquivo traz os
detalhes de execução do experimento principal (scripts 01 a 05).

Segue o Cap. 3 do TCC 2: 14 características stateless do CIC-Bell-DNS-EXF-2021,
cenários light e heavy separados, classe positiva = ataque.

| Arquivo | O que faz |
|---|---|
| `comum.py` | leitura dos dados, constantes, pipeline (OrdinalEncoder → SMOTE-NC → classificador) |
| `01_verificar_dados.py` | relatório de verificação (contagens × Tabela 1, ausentes, tipos, duplicatas, sld × classe) |
| `02_particao_desenvolvimento.py` | separa os 15% de desenvolvimento (estratificado, semente 42) e salva os índices |
| `03_experimento.py` | Grid Search (uma vez por configuração) + 5 sementes; grava cada execução no CSV; retomável |
| `04_analise.py` | tabelas média ± dp (CSV e LaTeX), curvas ROC/PR, importância das características |
| `executar_tudo.sh` | roda 03 e 04 em sequência, com até 3 tentativas |

Os scripts leem o `.zip` original diretamente (inclusive os zips internos).
Não é preciso extrair nada.

## Para instalar, verificar e rodar o piloto

```bash
# 1. Pacotes do sistema (Debian 13)
sudo apt install python3-venv tmux

# 2. Ambiente Python isolado (instalar AGORA, antes de ficar sem internet)
cd ~/tcc2_experimento
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Dados
mkdir -p dados
cp <pasta onde baixou>/CICBellEXFDNS2021.zip dados/

# 4. Verificação dos dados (~10 s) -> resultados/verificacao/relatorio_verificacao.txt
python 01_verificar_dados.py --dados dados/CICBellEXFDNS2021.zip

# 5. Partição de desenvolvimento (uma única vez; não é sobrescrita depois)
python 02_particao_desenvolvimento.py --dados dados/CICBellEXFDNS2021.zip

# 6. Teste rápido do código inteiro (4% dos dados, ~1 min) -> pasta resultados_teste/
python 03_experimento.py --dados dados/CICBellEXFDNS2021.zip --teste-rapido --n-jobs 6
python 04_analise.py --saida resultados_teste

# 7. Piloto: 1 configuração (heavy, Random Forest, SMOTE-NC, a mais cara), 1 semente
python 03_experimento.py --dados dados/CICBellEXFDNS2021.zip --piloto --n-jobs 6
python 04_analise.py --saida resultados/piloto
```

O piloto grava os resultados em `resultados/piloto/`. Os hiperparâmetros que ele
ajusta ficam em `resultados/hiperparametros/` e são reaproveitados pelo experimento
completo, porque o ajuste é determinístico. No log (`resultados/logs/experimento.log`),
veja o tempo do Grid Search e o da execução. Esse é o pior caso; as outras configurações
são mais rápidas.

**n_jobs:** o piloto fixa `n_jobs=6` em `resultados/ambiente.json`. A partir daí, o
script se recusa a rodar com outro valor, para que os tempos sejam comparáveis. Para
usar 12, apague a pasta `resultados/piloto` e o arquivo `resultados/ambiente.json` e
refaça o piloto com `--n-jobs 12`. 6 (núcleos físicos) é o recomendado.

**n_jobs_grid:** `--n-jobs-grid` define quantos processos o Grid Search usa em
paralelo; se for omitido, usa o mesmo valor de `--n-jobs`. Cada processo carrega a
própria cópia dos dados e do modelo, então reduzir esse número diminui o uso de memória.
Ele não afeta os tempos medidos nas execuções finais (que usam `--n-jobs`) e não é
verificado contra o `ambiente.json`; o valor usado fica registrado em `n_jobs_grid` no
JSON de cada configuração em `resultados/hiperparametros/`. Se o Grid Search estourar a
memória:

```bash
python 03_experimento.py --dados dados/CICBellEXFDNS2021.zip --piloto --n-jobs 6 --n-jobs-grid 3
N_JOBS_GRID=3 ./executar_tudo.sh       # no experimento completo
```

## Para rodar o experimento completo

```bash
cd ~/tcc2_experimento
tmux new -s tcc                      # sessão que sobrevive ao fechamento do terminal
systemd-inhibit --what=idle:sleep:handle-lid-switch --who=tcc \
    --why="experimento TCC" ./executar_tudo.sh
# Para sair do tmux sem parar nada: Ctrl+b e depois d
# Para voltar:                       tmux attach -t tcc
```

Para incluir o experimento extra (Random Forest sem a coluna `sld`):
`SEM_SLD=1 ./executar_tudo.sh` (na mesma linha do `systemd-inhibit`).

### Evitar suspensão

1. `systemd-inhibit` (acima) bloqueia suspensão por ociosidade, `systemctl suspend` e
   o fechamento da tampa enquanto o script roda. Confira em outro terminal:
   `systemd-inhibit --list` deve mostrar a linha "tcc".
2. Por garantia, desligue também no ambiente gráfico. No GNOME:
   Configurações → Energia → *Suspensão automática*: desligada.
   O bloqueio de tela pode ficar ligado, porque não interrompe o processo.
3. Alternativa radical, se ainda suspender:
   `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`.
   Ao terminar, desfaça com
   `sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target`.
4. Deixe o notebook na tomada e não faça logout (dormir a tela não faz mal).

### Acompanhar e retomar

```bash
tail -f resultados/logs/experimento.log
wc -l resultados/resultados_execucoes.csv      # 40 execuções + cabeçalho = 41 linhas
```

Se cair (falta de luz, Ctrl+C, reinício), rode o mesmo comando de novo. O que já está
no CSV é pulado, e os Grid Search já feitos são lidos dos JSON.

## Depois: análise

```bash
python 04_analise.py --saida resultados        # vírgula decimal (padrão)
python 04_analise.py --saida resultados --decimal ponto
```

Gera em `resultados/analise/`: `resumo_metricas.csv`, `tabela_metricas.tex`,
`tabela_matriz_confusao.tex`, `tabela_custo.tex` (requerem `\usepackage{graphicx}`),
`importancia_caracteristicas.csv` e as figuras em PNG e PDF.

## O que fica salvo

```
resultados/
  verificacao/                 relatório e tabulações (script 01)
  particoes/dev_<cenario>.csv  índices da partição de desenvolvimento (+ .json com hash)
  hiperparametros/<config>.json  escolhidos no Grid Search (+ cv_results/ completo)
  resultados_execucoes.csv     uma linha por (configuração, semente)
  curvas/                      pontos ROC e PR de cada execução (.csv.gz)
  importancias/                importância das características de cada execução
  ambiente.json                versões das bibliotecas, CPU, RAM, n_jobs
  logs/experimento.log
  piloto/                      mesmos arquivos, só do piloto
```

## Decisões de implementação

- **Dados:** os arquivos `benign_1` e `benign_2` (Benign.zip na raiz da pasta CSV)
  entram nos dois cenários. Só assim as contagens fecham com a Tabela 1.
- **"nan"/"null":** a leitura usa `keep_default_na=False`. Na leitura padrão do
  pandas, 31 valores de `longest_word` (as palavras "nan"/"null" de domínios como
  `nanit`) virariam ausentes falsos. A troca de ausentes por zero continua no código,
  mas não altera nenhuma linha.
- **Duplicatas** não são removidas (a metodologia não prevê); são só relatadas.
- **PR-AUC** = precisão média (`average_precision_score`), como na Seção 3.6.
- **Tempo de treino** = ajuste do pipeline menos o tempo do SMOTE-NC (inclui o
  OrdinalEncoder, que leva frações de segundo). **Tempo de predição** = `predict()` no
  teste; o `predict_proba()` (usado nas curvas) é cronometrado à parte.
- **n_jobs:** no Grid Search, as combinações rodam em paralelo em `n_jobs_grid`
  processos (cada ajuste com 1 núcleo). Nas execuções finais, o Random Forest usa
  `n_jobs` núcleos; a Árvore de Decisão é sequencial por natureza.
- **SMOTE-NC:** k = 5 vizinhos, `sampling_strategy=1.0` (1:1), colunas categóricas
  indicadas por posição após o encoder.
- **Empates no Grid Search:** vence a primeira combinação da grade (comportamento do
  scikit-learn). O log avisa quando isso acontece.
- **Sementes:** partição dev = 42; validação cruzada = 42; repetições = 0 a 4 (a mesma
  semente controla a divisão 70/30, o SMOTE-NC e o modelo).
- **Importância:** salva para RF e DT; a figura mostra só o RF.
