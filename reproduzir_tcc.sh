#!/usr/bin/env bash
# Reproduz todos os resultados do TCC 2, na ordem, e gera as tabelas e figuras
# do Capítulo 4 em resultados/tcc/.
#
#   01  verificação dos dados .................. Seção 4.2 (Tabela 4)
#   02  partição de desenvolvimento (15%) ...... Seção 3.3
#   03  experimento principal + RF sem sld ..... Seções 4.1, 4.3 e 4.8 (Tabelas 3, 5, 6, 7, 11)
#   04  análise do experimento principal
#   06  experimentos complementares ............ Seções 4.4 a 4.9 (Tabelas 8, 9, 10; Figuras 3 a 6)
#   07  análise dos complementares
#   08  tabelas e figuras no formato do TCC .... resultados/tcc/
#
# Tempo total aproximado: algumas horas com n_jobs=6 (Ryzen 5 2600, 11,6 GB).
# Retomável: se for interrompido, rode de novo e o que já terminou é pulado.
#
# Uso (dados em dados/CICBellEXFDNS2021.zip; ver README.md):
#   ./reproduzir_tcc.sh
# Variáveis opcionais: DADOS=caminho/do/zip  N_JOBS=6  N_JOBS_GRID=3
set -eu
cd "$(dirname "$0")"
DADOS="${DADOS:-dados/CICBellEXFDNS2021.zip}"
export DADOS
export N_JOBS="${N_JOBS:-6}"   # o mesmo n_jobs em todas as execuções (Seção 3.5)
export N_JOBS_GRID="${N_JOBS_GRID:-$N_JOBS}"

if [ ! -e "$DADOS" ]; then
    echo "Não encontrei $DADOS. Baixe o CIC-Bell-DNS-EXF-2021 (ver README.md)." >&2
    exit 2
fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

python 01_verificar_dados.py --dados "$DADOS" --saida resultados
python 02_particao_desenvolvimento.py --dados "$DADOS" --saida resultados
SEM_SLD=1 ./executar_tudo.sh          # 03 (com o RF sem sld da Seção 4.8) + 04
./executar_complementares.sh          # 06 + 07
python 08_tabelas_figuras_tcc.py --saida resultados
