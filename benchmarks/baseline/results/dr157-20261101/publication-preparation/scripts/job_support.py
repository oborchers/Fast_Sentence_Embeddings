import os,time,shutil
from pathlib import Path
from contextlib import contextmanager
COORD=Path('/private/tmp/claude-501/-Users-oliverborchers-Desktop-Code-nosync-Fast-Sentence-Embeddings/92594212-c3b7-4241-9bf5-4a2ced3b53cf/scratchpad')
_next=0.0
@contextmanager
def reservation():
 global _next
 lock=COORD/'heavy.lock'
 identity=f'dr157-hub pid={os.getpid()} start={time.time()}'
 while True:
  paused=any((COORD/n).exists() for n in ('pause-local.marker','timing-window.marker'))
  if paused or time.monotonic()<_next:
   time.sleep(15);continue
  try: lock.mkdir()
  except FileExistsError:
   time.sleep(15);continue
  owner=lock/'owner';owner.write_text(identity)
  if any((COORD/n).exists() for n in ('pause-local.marker','timing-window.marker')):
   owner.unlink();lock.rmdir();time.sleep(15);continue
  break
 try:
  if shutil.disk_usage(Path.home()/'fse-dr157').free <25*2**30: raise RuntimeError('Less than 25 GiB free')
  yield
 finally:
  if owner.read_text()==identity:
   owner.unlink();lock.rmdir();_next=time.monotonic()+120
