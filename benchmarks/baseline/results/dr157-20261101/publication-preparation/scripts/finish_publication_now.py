"""Observe existing uploads, then run the explicitly requested full fresh readback."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub')
OUT = Path.home() / 'fse-dr157/publication-now'
FINAL = OUT / 'final-readback'
STATE = OUT / 'completion-monitor.json'
UPLOAD_PID = 45458
sys.path.insert(0, str(ROOT / 'benchmarks'))
from hub_readback_evidence import write_atomic
from hub_publication import file_sha256
EXPECTED = 'b569e35ee442263a3b73fb3d5f8ec89cf36488816bb36af3426e5caef116a436'
state = {'status': 'observing-upload', 'monitor_pid': os.getpid(), 'upload_pid': UPLOAD_PID, 'milestones_notified': ['Upload mit erstem frischen Readback:5'], 'manifest_sha256': EXPECTED}


def save():
    write_atomic(STATE, state)


def notify(target, message):
    resolved = subprocess.run(['herdr', 'agent', 'get', target], capture_output=True, text=True, timeout=20)
    if resolved.returncode != 0:
        raise RuntimeError('cannot resolve notification target')
    agent = json.loads(resolved.stdout)['result']['agent']
    if agent['name'] != target:
        raise ValueError('notification target identity differs')
    sent = subprocess.run(['herdr', 'agent', 'prompt', target, message], capture_output=True, text=True, timeout=20)
    if sent.returncode != 0:
        raise RuntimeError('notification submission failed')


def progress(count, label):
    for number in range(5, 31, 5):
        key = f'{label}:{number}'
        if count >= number and key not in state['milestones_notified']:
            notify('fse-main', f'dr157-hub: {label}: {number}/30 Repos erfolgreich geprüft. Modelle bleiben unverändert normalisiert. Belege unter {OUT}. Kein Merge.')
            state['milestones_notified'].append(key)
            save()


def main():
    if STATE.exists() or FINAL.exists():
        raise FileExistsError('completion evidence already exists; refusing overwrite')
    save()
    while True:
        ledger = json.loads((OUT / 'publication-ledger.json').read_bytes())
        count = sum(row['status'] == 'verified' for row in ledger['repositories'])
        state['uploaded_and_verified'] = count
        save()
        progress(count, 'Upload mit erstem frischen Readback')
        if ledger['status'] == 'stopped':
            raise RuntimeError('upload stopped; preserved ledger identifies failure')
        if ledger['status'] == 'complete':
            if count != 30:
                raise ValueError('incomplete verified repository set')
            break
        os.kill(UPLOAD_PID, 0)
        time.sleep(10)
    if file_sha256(OUT / 'publication-manifest.json') != EXPECTED:
        raise ValueError('manifest changed before final readback')
    state['status'] = 'final-fresh-readback-running'
    save()
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'benchmarks/verify_published_hub.py'), '--manifest', str(OUT / 'publication-manifest.json'), '--upload-receipts', str(OUT / 'upload-receipts.json'), '--output', str(FINAL), '--scratch', str(OUT / 'final-fresh-cache')]
    with (OUT / 'final-readback-progress.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state['final_readback_pid'] = process.pid
        save()
        while process.poll() is None:
            evidence_path = FINAL / 'evidence.json'
            if evidence_path.exists():
                evidence = json.loads(evidence_path.read_bytes())
                state['final_checked'] = len(evidence['repositories'])
                save()
                progress(state['final_checked'], 'Zusätzliche Abschlussprüfung')
            time.sleep(10)
    evidence = json.loads((FINAL / 'evidence.json').read_bytes())
    if process.returncode != 0 or not evidence['complete'] or not evidence['passed'] or len(evidence['repositories']) != 30:
        raise RuntimeError('final readback failed; preserve evidence and do not repair Hub')
    state.update(status='complete', final_checked=30, evidence_sha256=file_sha256(FINAL / 'evidence.json'))
    save()
    progress(30, 'Zusätzliche Abschlussprüfung')
    message = f'dr157-hub: Alle 30 öffentlichen Uploads und die zusätzliche vollständige Abschlussprüfung sind erfolgreich. Beleg: {FINAL / "evidence.json"}, SHA256 {state["evidence_sha256"]}. Tabelle: {FINAL / "report.md"}. Bitte Abschlussbelege auswerten und wie vereinbart dauerhaft archivieren. Keine Hub-Reparaturen und kein Merge.'
    notify('fse-main', message)
    notify('dr157-hub', message)


if __name__ == '__main__':
    with (OUT / 'completion-monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as error:
            state.update(status='stopped', error_type=type(error).__name__)
            save()
            message = f'dr157-hub: Abschlussüberwachung gestoppt ({type(error).__name__}). Bitte {STATE} und die Upload-/Readback-Belege prüfen. Nichts auf dem Hub löschen oder überschreiben.'
            for target in ['fse-main', 'dr157-hub']:
                try:
                    notify(target, message)
                except Exception:
                    pass
            raise SystemExit(1) from None
