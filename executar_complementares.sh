#!/usr/bin/env bash
# Roda os experimentos complementares (06) e depois a análise deles (07):
# Seções 4.4 a 4.9 do TCC 2.
# Não altera nada do experimento principal. Retomável: rode de novo se cair.
#
# Uso (dentro do tmux, impedindo a suspensão):
#   systemd-inhibit --what=idle:sleep:handle-lid-switch --who=tcc --why="complementares TCC" ./executar_complementares.sh
#
# Variáveis opcionais:
#   DADOS=dados/CICBellEXFDNS2021.zip  N_JOBS=6
#   SEM_DUPLICATAS=1   (inclui a variante opcional sem linhas duplicadas)
#   EXPERIMENTOS="referencia grupo"   (só estes; padrão: todos)
set -u
cd "$(dirname "$0")"
DADOS="${DADOS:-dados/CICBellEXFDNS2021.zip}"
N_JOBS="${N_JOBS:-6}"
EXTRA=""
for e in ${EXPERIMENTOS:-referencia grupo permutacao balanceamento cruzado}; do
    EXTRA="$EXTRA --experimento $e"
done
[ "${SEM_DUPLICATAS:-0}" = "1" ] && EXTRA="$EXTRA --experimento sem_duplicatas"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

for tentativa in 1 2 3; do
    echo "=== Tentativa $tentativa: $(date) ==="
    python 06_complementares.py --dados "$DADOS" --saida resultados --n-jobs "$N_JOBS" $EXTRA
    status=$?
    [ $status -eq 0 ] && break
    [ $status -eq 2 ] && { echo "Erro de configuração (ver log). Abortando."; exit 2; }
    sleep 60
done

python 07_analise_complementares.py --saida resultados
echo "=== Fim: $(date) (status: $status) ==="
exit $status
