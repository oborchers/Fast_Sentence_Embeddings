"""Explicitly authorized publication preparation without asserting a package release."""
import copy
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub')
sys.path.insert(0, str(ROOT / 'benchmarks'))
from hub_publication import card_example, file_sha256, file_table, sha256_bytes, size_tables
from hub_release_cards import DEVELOPMENT_NOTICE, PINNED_INSTALL, check_files, release_entry, write_card
from hub_readback_evidence import mkdir_durable, write_atomic
from job_support import reservation
import fse

BASE_SHA = 'ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69'
BASIS = ROOT / 'benchmarks/baseline/dr157/publication-manifest.json'
OUTPUT = Path.home() / 'fse-dr157/publication-now'


def render(text, repo):
    if text.count(PINNED_INSTALL) != 1 or text.count(DEVELOPMENT_NOTICE) != 1:
        raise ValueError('unexpected basis card pattern')
    result = text.replace(PINNED_INSTALL, 'pip install "fse>=2.0"').replace(
        DEVELOPMENT_NOTICE, 'This model requires fse 2.0 or later.')
    result, count = re.subn(r'(?m)^- written by: fse[^\r\n]*\r?\n', '', result)
    if count != 1 or card_example(result, repo) != card_example(text, repo):
        raise ValueError('writer removal or literal Python example differs')
    if any(term in result.lower() for term in ('git+https', 'in development', 'written by:', 'pod')):
        raise ValueError('forbidden card text remains')
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError('publication output already exists; never overwrite it')
    if file_sha256(BASIS) != BASE_SHA:
        raise ValueError('basis manifest SHA256 differs')
    original = json.loads(BASIS.read_bytes())
    if len(original['repositories']) != 30:
        raise ValueError('expected all 30 repositories')
    plans = []
    with reservation():
        for entry in original['repositories']:
            folder = Path(entry['local_folder'])
            check_files(folder, entry['files'])
            old = (folder / 'README.md').read_bytes().decode('utf-8')
            plans.append((entry, old, render(old, entry['repo'])))
        mkdir_durable(OUTPUT)
        mkdir_durable(OUTPUT / 'models')
        mkdir_durable(OUTPUT / 'cards')
        result = copy.deepcopy(original)
        result['repositories'] = []
        for entry, old, new in plans:
            after = release_entry(entry, new, OUTPUT)
            changed = [name for name in entry['files'] if entry['files'][name] != after['files'][name]]
            if changed != ['README.md']:
                raise ValueError('non-README payload changed')
            result['repositories'].append(after)
            if entry['repo'] == 'fse/paranmt-300-sif10':
                write_card(OUTPUT / 'paranmt-300-sif10.diff', ''.join(difflib.unified_diff(
                    old.splitlines(keepends=True), new.splitlines(keepends=True),
                    fromfile='previous/README.md', tofile='publication/README.md')))
        result['total_bytes'] = sum(e['total_bytes'] for e in result['repositories'])
        result['approved'] = True
        result['approval_gate'] = 'Explicit Upload frei authorization. Create new public repositories only; stop on existing names.'
        result['publication_preparation'] = {
            'mode': 'publication-now-without-package-release',
            'basis_manifest_sha256': BASE_SHA,
            'actual_runtime_version': fse.__version__,
            'install_requirement': 'fse>=2.0',
            'writer_version_line': 'omitted',
        }
        if 'pod' in json.dumps(result).lower():
            raise ValueError('forbidden text in publication manifest')
        path = OUTPUT / 'publication-manifest.json'
        write_atomic(path, result)
        receipt = {
            'basis_manifest_sha256': BASE_SHA,
            'publication_manifest_sha256': file_sha256(path),
            'repositories': 30, 'files': sum(len(e['files']) for e in result['repositories']),
            'only_changed_payload_file': 'README.md',
            'all_model_sha256_unchanged': True, 'hub_writes_performed': False,
        }
        write_atomic(OUTPUT / 'preparation-receipt.json', receipt)
        write_card(OUTPUT / 'publication-inventory.md', '\n'.join([
            '# Authorized PUBLIC publication inventory', '', *size_tables(result), *file_table(result), '']))
        print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1) from None
