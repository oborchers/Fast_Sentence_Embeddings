import json,sys
from pathlib import Path
from hub_publication_auth import client,safe_error
ROOT=Path('/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr157-hub')
sys.path.insert(0,str(ROOT/'benchmarks'))
from hub_readback_evidence import write_atomic
from huggingface_hub.errors import RepositoryNotFoundError
out=Path.home()/'fse-dr157/name-preflight-now.json'
try:
    api=client()
    api.whoami()
    manifest=json.loads((ROOT/'benchmarks/baseline/dr157/publication-manifest.json').read_bytes())
    rows=[]
    for entry in manifest['repositories']:
        repo=entry['repo']
        try:
            info=api.repo_info(repo,repo_type='model')
            rows.append({'repo':repo,'status':'name_conflict'})
        except RepositoryNotFoundError as error:
            if getattr(error.response,'status_code',None)!=404: raise
            rows.append({'repo':repo,'status':'absent'})
    result={'credential_source':'approved 1Password mount','hub_writes_performed':False,'repositories':rows,'names_clear':all(r['status']=='absent' for r in rows)}
    write_atomic(out,result)
    print(json.dumps({'checked':len(rows),'names_clear':result['names_clear'],'conflicts':[r['repo'] for r in rows if r['status']=='name_conflict']}),flush=True)
    raise SystemExit(0 if result['names_clear'] else 1)
except Exception as error:
    print(json.dumps({'stage':'credential_or_name_preflight','error':safe_error(error),'hub_writes_performed':False}),flush=True)
    raise SystemExit(1) from None
