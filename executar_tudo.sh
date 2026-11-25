#!/usr/bin/env bash
# Roda o experimento principal (03) e depois a análise (04): Seções 4.1 e 4.3 do TCC 2.
# Com SEM_SLD=1, inclui o Random Forest sem a coluna sld (Seção 4.8, Tabela 11).
# Para reproduzir o TCC inteiro, use ./reproduzir_tcc.sh.
# Retomável: se cair ou for interrompido, rode de novo e ele continua de onde parou.
#
# Uso (dentro do tmux, impedindo a suspensão):
#   systemd-inhibit --what=idle:sleep:handle-lid-switch --who=tcc --why="experimento TCC" ./executar_tudo.sh
#
# Variáveis opcionais:
#   DADOS=dados/CICBellEXFDNS2021.zip  N_JOBS=6  SEM_SLD=1 (inclui o RF sem sld)
#   N_JOBS_GRID=3 (processos do Grid Search; padrão = N_JOBS)
set -u
cd "$(dirname "$0")"
DADOS="${DADOS:-dados/CICBellEXFDNS2021.zip}"
N_JOBS="${N_JOBS:-6}"
N_JOBS_GRID="${N_JOBS_GRID:-$N_JOBS}"
EXTRA=""
[ "${SEM_SLD:-0}" = "1" ] && EXTRA="--sem-sld"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

# Até 3 tentativas: execuções que falharem são refeitas; as concluídas são puladas.
for tentativa in 1 2 3; do
    echo "=== Tentativa $tentativa: $(date) ==="
    python 03_experimento.py --dados "$DADOS" --saida resultados --n-jobs "$N_JOBS" --n-jobs-grid "$N_JOBS_GRID" $EXTRA
    status=$?
    [ $status -eq 0 ] && break
    [ $status -eq 2 ] && { echo "Erro de configuração (ver log). Abortando."; exit 2; }
    sleep 60
done

python 04_analise.py --saida resultados
echo "=== Fim: $(date) (status do experimento: $status) ==="
exit $status
