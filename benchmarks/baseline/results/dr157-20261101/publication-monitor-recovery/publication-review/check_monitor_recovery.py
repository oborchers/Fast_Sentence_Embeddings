"""Fault-injection checks for publication process guards; never contact Hub or Herdr."""
import contextlib
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from unittest.mock import patch

SCRIPTS=Path('/Users/oliverborchers/fse-dr157/scripts')
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('monitor_under_test',SCRIPTS/'finish_publication_now.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
results=[]
with tempfile.TemporaryDirectory(prefix='monitor-check-',dir='/Users/oliverborchers/fse-dr157/publication-review') as temporary:
 home=Path(temporary)
 out=home/'fse-dr157/publication-now'
 out.mkdir(parents=True)
 existing={'status':'complete','evidence_sha256':'preserve-existing-proof'}
 state=out/'completion-monitor.json'
 state.write_text(json.dumps(existing))
 before=state.read_bytes()
 with patch.object(Path,'home',return_value=home),patch.object(sys,'argv',['monitor']),contextlib.redirect_stdout(io.StringIO()) as output:
  try:runpy.run_path(str(SCRIPTS/'finish_publication_now.py'),run_name='__main__')
  except SystemExit as error:assert error.code==1
  else:raise AssertionError('existing state was accepted')
 assert state.read_bytes()==before
 assert json.loads(output.getvalue())['existing_evidence_unchanged'] is True
 results.append('existing_monitor_evidence_preserved')
 m.STATE=out/'fault-injection-state.json'
 m.owns_state=True
 m.state={'status':'observing-upload','milestones_notified':[]}
 def timeout(*_args):raise TimeoutError('injected notification failure')
 with patch.object(m,'notify',side_effect=timeout):m.progress(5,'test')
 assert m.state['status']=='observing-upload' and m.state['milestones_notified']==[]
 assert m.state['notification_errors']['test:5']=='TimeoutError'
 with patch.object(m,'notify') as delivered:m.progress(5,'test')
 assert delivered.call_count==1 and m.state['milestones_notified']==['test:5']
 assert not m.state['notification_errors']
 results.append('notification_failure_does_not_stop_observer_and_retries')
 lock=out/'publication.lock'
 with lock.open('a') as held:
  fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
  code='import pathlib,runpy,sys;from unittest.mock import patch;sys.path.insert(0,sys.argv[1]);\nwith patch.object(pathlib.Path,"home",return_value=pathlib.Path(sys.argv[2])):runpy.run_path(sys.argv[1]+"/publish_hub_now.py",run_name="__main__")'
  child=subprocess.run([sys.executable,'-c',code,str(SCRIPTS),str(home)],capture_output=True,text=True,timeout=30)
  assert child.returncode==1,(child.returncode,child.stderr)
  refusal=json.loads(child.stdout)
  assert refusal['reason']=='another publisher holds the process lock'
  assert state.read_bytes()==before
 results.append('competing_publisher_refused_without_evidence_writes')
 resume=out/'recover.json'
 previous={'status':'observing-upload','monitor_pid':99999999,'upload_pid':m.UPLOAD_PID,'manifest_sha256':m.EXPECTED,'milestones_notified':['preserved:5']}
 resume.write_text(json.dumps(previous))
 m.STATE=resume;m.FINAL=out/'missing-final';m.owns_state=False
 with patch.object(sys,'argv',['monitor','--resume-observer']),patch.object(m.os,'kill',side_effect=ProcessLookupError):m.claim_state()
 assert m.state['milestones_notified']==['preserved:5'] and m.state['prior_monitor_pid']==99999999
 assert m.owns_state is True and m.state['recovered_observer'] is True
 results.append('explicit_observer_recovery_preserves_history')
 final=out/'probe-final';final.mkdir()
 (final/'evidence.json').write_text(json.dumps({'repositories':[{'status':'passed'}]*4+[{'status':'failed'}],'complete':False,'passed':False}))
 (out/'publication-ledger.json').write_text(json.dumps({'status':'complete','repositories':[{'status':'verified'}]*30}))
 class FakeProcess:
  pid=123
  returncode=1
  def __init__(self):self.calls=0
  def poll(self):
   self.calls+=1
   return None if self.calls==1 else 1
 process=FakeProcess();counts=[]
 m.OUT=out;m.FINAL=final;m.STATE=out/'runtime-state.json';m.state={'milestones_notified':[]}
 with patch.object(m,'claim_state'),patch.object(m,'file_sha256',return_value=m.EXPECTED),patch.object(m.subprocess,'Popen',return_value=process),patch.object(m.time,'sleep'),patch.object(m,'progress',side_effect=lambda count,label:counts.append((count,label))):
  try:m.main()
  except RuntimeError as error:assert 'final readback failed' in str(error)
  else:raise AssertionError('failed readback accepted')
 assert (4,'Zusätzliche Abschlussprüfung') in counts
 assert (5,'Zusätzliche Abschlussprüfung') not in counts
 results.append('failed_readbacks_never_counted_as_successes')
print(json.dumps({'checks':results,'passed':len(results),'hub_writes_performed':False,'herdr_messages_sent':False},indent=2))
