from fse import IndexedList

import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")

from joblib import load
from fse.models import Average
sentences = load("sentences.joblib")


s = IndexedList([sent.split() for sent in sentences], pre_splitted=True)

# from profiling.tracing import TracingProfiler
# profiler = TracingProfiler()
# profiler.start()


# import cProfile
 
# pr = cProfile.Profile()
# pr.enable()

model = Average(glove, workers=1)

model.train(s)

# # pr.disable()
 
# # pr.print_stats(sort='time')

# profiler.stop()
# profiler.run_viewer()