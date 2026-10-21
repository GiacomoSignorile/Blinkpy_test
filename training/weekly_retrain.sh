#!/usr/bin/env bash
# Weekly: auto-label the new people crops and retrain on the GPU server, then bring the results home.
#   - skips itself when fewer than MIN_NEW crops are unscored (nothing worth a 45 min GPU job)
#   - imports the new auto-labels into the database (auto_* columns only; hand labels are never touched)
#   - saves the model as data/models/persona_clipL_auto_<date>.pt but does NOT switch production:
#     compare the validation numbers in the log with v3, then set PERSON_MODEL in docker-compose.yml.
# Runs hourly from cron but acts at most once a week: when a week has passed, enough new crops exist and the
# GPU server (RTX 5070) is reachable. If the server is off it simply waits and checks again an hour later.
# Usage:  training/weekly_retrain.sh [--check] [--force]      (cron runs it without arguments)
set -euo pipefail

ROOT=/home/giacomo/Blinkpy_test
SERVER=giacomo@100.112.122.139
RDIR=blinkwatch-train
MIN_NEW=${MIN_NEW:-50}
PY="$ROOT/.venv/bin/python"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 $SERVER"
STAMP=$(date +%Y%m%d)
EXPORT="export_$STAMP"
BASELINE="v3: group 75.9%, time 78.1%"
log() { echo "$(date '+%F %T') $*"; }

cd "$ROOT"
exec 9>"$ROOT/data/.retrain.lock"
flock -n 9 || { log "already running, exiting"; exit 0; }

OK_FILE="$ROOT/data/.retrain_last_ok"; FAIL_FILE="$ROOT/data/.retrain_fails"
VERBOSE=0; [[ "${1:-}" == "--check" ]] && VERBOSE=1
FORCE=0; [[ "${1:-}" == "--force" ]] && FORCE=1
say() { if [[ "$VERBOSE" == 1 ]]; then log "$*"; fi; }   # hourly runs stay silent unless they actually do something

# data/.retrain_not_before holds an epoch time: nothing starts before it (used to schedule the next run on a date)
NOT_BEFORE=$(cat "$ROOT/data/.retrain_not_before" 2>/dev/null || echo 0)
if [[ "$FORCE" == 0 && $(date +%s) -lt "$NOT_BEFORE" ]]; then say "scheduled for $(date -d @"$NOT_BEFORE" '+%F %T'), not yet"; exit 0; fi

# runs every hour from cron; only acts when a weekly run is due, there is work to do and the GPU server is up
LAST=$(cat "$OK_FILE" 2>/dev/null || echo 0)
if [[ "$FORCE" == 0 && $(( $(date +%s) - LAST )) -lt $(( 7 * 86400 )) ]]; then say "last successful run is under 7 days old"; exit 0; fi
FAILS=$(cat "$FAIL_FILE" 2>/dev/null || echo 0)
if [[ "$FORCE" == 0 && "$FAILS" -ge 3 ]]; then say "3 failed attempts in a row: waiting for a manual run (--force)"; exit 0; fi

NEW=$("$PY" - <<'E'
import sqlite3
d = sqlite3.connect("file:data/events.db?mode=ro", uri=True)
print(d.execute("select count(*) from crops where kind='persona' and label is null and auto_label is null").fetchone()[0])
E
)
say "unscored people crops: $NEW (threshold $MIN_NEW)"
if [[ "$NEW" -lt "$MIN_NEW" && "$FORCE" == 0 ]]; then say "not enough new crops"; exit 0; fi

if ! $SSH true 2>/dev/null; then say "GPU server (RTX 5070) is not up yet, will check again in an hour"; exit 0; fi
if [[ "$VERBOSE" == 1 ]]; then log "a run would start now"; exit 0; fi

fail() { echo $(( FAILS + 1 )) > "$FAIL_FILE"; log "FAILED ($*), attempt $(( FAILS + 1 )) of 3"; exit 1; }
trap 'fail "unexpected error on line $LINENO"' ERR

log "exporting labelled crops -> data/$EXPORT"
"$PY" training/export_dataset.py "data/$EXPORT" > /dev/null
(cd data && tar czf "$EXPORT.tgz" "$EXPORT")
scp -q "data/$EXPORT.tgz" "data/${EXPORT}_manifest.csv" "$SERVER:$RDIR/"
scp -q training/{autolabel_ft,train_final,autolabel,finetune,benchmark_persons}.py "$SERVER:$RDIR/"
$SSH "cd ~/$RDIR && tar xzf $EXPORT.tgz && rm -f autolabel_ft.done autolabel_auto.csv"

log "starting autolabel + retrain on the server"
$SSH "cd ~/$RDIR && setsid nohup bash -c 'source .venv/bin/activate && python autolabel_ft.py --data $EXPORT --oof oof_persona.npz --out persona_clipL_auto.pt --csv autolabel_auto.csv > autolabel_auto.log 2>&1' >/dev/null 2>&1 < /dev/null &"

for _ in $(seq 1 180); do   # up to 3 hours, one check a minute
    sleep 60
    if $SSH "test -f ~/$RDIR/autolabel_ft.done" 2>/dev/null; then DONE=1; break; fi
    if ! $SSH "pgrep -f '[a]utolabel_ft.py --data'" >/dev/null 2>&1; then
        $SSH "grep -v -i warn ~/$RDIR/autolabel_auto.log | tail -5" || true; fail "server job stopped without finishing"
    fi
done
[[ "${DONE:-0}" == 1 ]] || fail "timed out waiting for the server"

log "server finished, bringing results home"
$SSH "grep -v -i warn ~/$RDIR/autolabel_auto.log | grep -E 'accepted|mean|final model'" || true
scp -q "$SERVER:$RDIR/autolabel_auto.csv" "data/autolabel_persona_auto.csv"
scp -q "$SERVER:$RDIR/persona_clipL_auto.pt" "data/models/persona_clipL_auto_$STAMP.pt"
DATA_DIR=data "$PY" training/import_autolabels.py data/autolabel_persona_auto.csv | head -1
log "baseline to beat: $BASELINE. New model saved as data/models/persona_clipL_auto_$STAMP.pt (production NOT switched)"

# tidy up: this run's export, older exports on the server, all but the 2 newest automatic models
rm -rf "data/$EXPORT" "data/$EXPORT.tgz" "data/${EXPORT}_manifest.csv"
$SSH "cd ~/$RDIR && rm -f $EXPORT.tgz && ls -d export_20* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf && ls export_*_manifest.csv 2>/dev/null | sort | head -n -2 | xargs -r rm -f"
ls -1t data/models/persona_clipL_auto_*.pt 2>/dev/null | tail -n +3 | xargs -r rm -f
date +%s > "$OK_FILE"; rm -f "$FAIL_FILE"
log "done"
