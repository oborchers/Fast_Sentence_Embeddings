"""Execute the explicit Upload frei instruction, new public repositories only."""
import json
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from hub_publication_auth import client, safe_error
ROOT = Path('/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub')
sys.path.insert(0, str(ROOT / 'benchmarks'))
from hub_publication import file_sha256
from hub_release_cards import check_files
from hub_readback_evidence import mkdir_durable, write_atomic
from verify_published_hub import checked_run, table
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import RepositoryNotFoundError

OUTPUT = Path.home() / 'fse-dr157/publication-now'
MANIFEST = OUTPUT / 'publication-manifest.json'
EXPECTED = 'b569e35ee442263a3b73fb3d5f8ec89cf36488816bb36af3426e5caef116a436'
LEDGER = OUTPUT / 'publication-ledger.json'
state = {}
phase = 'preflight'
current = None


def persist():
    write_atomic(LEDGER, state)
    uploaded = [r for r in state['repositories'] if r.get('revision')]
    write_atomic(OUTPUT / 'upload-receipts.json', {
        'publication_manifest_sha256': EXPECTED,
        'approved_repositories': state['approved_repositories'],
        'repositories': [{'repo': r['repo'], 'revision': r['revision']} for r in uploaded],
        'complete': len(uploaded) == 30,
        'hub_writes_performed': any(r.get('created_new') for r in state['repositories']),
    })
    checks = [r['readback'] for r in state['repositories'] if 'readback' in r]
    write_atomic(OUTPUT / 'readback-evidence.json', {
        'publication_manifest_sha256': EXPECTED,
        'repositories': checks,
        'complete': len(checks) == 30,
        'passed': len(checks) == 30 and all(r['status'] == 'passed' for r in checks),
    })
    from hub_release_cards import write_card
    report = OUTPUT / 'readback-report.md'
    # Derived report is regenerated; JSON receipts are atomically secured first.
    report.write_text('# Fresh public repository verification\n\n' + '\n'.join(table(checks)) + '\n', encoding='utf-8', newline='\n')


def require_absent(api, repo):
    try:
        api.repo_info(repo, repo_type='model')
    except RepositoryNotFoundError as error:
        if getattr(error.response, 'status_code', None) == 404:
            return
        raise
    raise FileExistsError('repository name already exists; no overwrite is permitted')


def main():
    global state, phase, current
    if LEDGER.exists():
        raise FileExistsError('existing publication ledger; do not restart or overwrite')
    if file_sha256(MANIFEST) != EXPECTED:
        raise ValueError('publication manifest changed')
    manifest = json.loads(MANIFEST.read_bytes())
    entries = sorted(manifest['repositories'], key=lambda e: (e['total_bytes'], e['repo']))
    if len(entries) != 30 or len({e['repo'] for e in entries}) != 30 or any(not e['repo'].startswith('fse/') for e in entries):
        raise ValueError('publication must contain exactly the approved 30 fse repositories')
    api = client()
    api.whoami()
    for entry in entries:
        require_absent(api, entry['repo'])
    state = {
        'publication_manifest_sha256': EXPECTED,
        'credential_source': 'approved 1Password mount only',
        'authorization': 'explicit Upload frei instruction',
        'approved_repositories': [e['repo'] for e in entries],
        'repositories': [], 'status': 'running',
    }
    mkdir_durable(OUTPUT / 'readback')
    mkdir_durable(OUTPUT / 'fresh-cache')
    persist()
    for index, entry in enumerate(entries, 1):
        current = entry['repo']
        phase = 'local-file-verification'
        if file_sha256(MANIFEST) != EXPECTED:
            raise ValueError('publication manifest changed during execution')
        check_files(Path(entry['local_folder']), entry['files'])
        if shutil.disk_usage(OUTPUT).free < 25 * 2**30:
            raise RuntimeError('less than 25 GiB free')
        require_absent(api, current)
        row = {'repo': current, 'status': 'creating', 'local_files_verified': True}
        state['repositories'].append(row)
        persist()
        phase = 'create-public-repository'
        api.create_repo(current, repo_type='model', private=False, exist_ok=False)
        row.update(created_new=True, visibility='public', status='created')
        persist()
        phase = 'verify-new-empty-repository'
        info = api.model_info(current, files_metadata=True)
        names = {item.rfilename for item in info.siblings}
        if info.id != current or info.private is not False or not names <= {'.gitattributes'} or not re.fullmatch(r'[a-f0-9]{40}', info.sha or ''):
            raise ValueError('new repository identity or initial tree is unexpected')
        row['initial_revision'] = info.sha
        row['status'] = 'uploading'
        persist()
        phase = 'upload-initial-payload'
        # The only possible replaced file is the automatic attributes in our newly created repository.
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=Path(entry['local_folder']) / name) for name in entry['files']]
        commit = api.create_commit(current, repo_type='model', operations=operations,
            commit_message='Publish prepared classic static embeddings',
            parent_commit=info.sha, num_threads=2)
        if not re.fullmatch(r'[a-f0-9]{40}', commit.oid):
            raise ValueError('upload returned no exact commit revision')
        row.update(revision=commit.oid, status='uploaded')
        persist()
        phase = 'fresh-public-readback'
        attempt_dir = OUTPUT / 'readback' / current.replace('/', '--')
        mkdir_durable(attempt_dir)
        args = SimpleNamespace(manifest=MANIFEST, scratch=OUTPUT / 'fresh-cache', output=attempt_dir, timeout=1800)
        checked = checked_run(entry, commit.oid, args)
        row['readback'] = checked
        row['status'] = 'verified' if checked['status'] == 'passed' else 'readback-failed'
        persist()
        if checked['status'] != 'passed':
            raise RuntimeError('public readback failed; no remote repair is attempted')
        print(json.dumps({'completed': index, 'total': 30, 'repo': current, 'revision': commit.oid, 'readback': 'passed'}), flush=True)
        if index % 5 == 0:
            print(json.dumps({'milestone': index, 'uploaded_and_verified': index}), flush=True)
    state['status'] = 'complete'
    persist()
    print(json.dumps({'status': 'complete', 'uploaded': 30, 'verified': 30, 'manifest_sha256': EXPECTED}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if state:
            state.update(status='stopped', failure={'repo': current, 'phase': phase, **safe_error(error)})
            persist()
        print(json.dumps({'status': 'stopped', 'repo': current, 'phase': phase, 'error': safe_error(error), 'remote_deletes_or_repairs': False}), flush=True)
        raise SystemExit(1) from None
