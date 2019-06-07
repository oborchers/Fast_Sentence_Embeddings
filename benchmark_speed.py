import numpy as np
from re import sub
import logging
import sys
import pathlib

np.random.seed(42)
logger = logging.getLogger(__name__)

# Import Sentence2Vec model and check if cython compiliation worked
from fse.models import Sentence2Vec
from fse.models.sentence2vec import CY_ROUTINES as CY_ROUTINES_TRAIN

# Import python sif-implementations
from fse.exp.sif_variants import sif_embeddings, \
	sif_embeddings_1, sif_embeddings_2, sif_embeddings_3, \
	sif_embeddings_4, sif_embeddings_5

# Import cython sif-implementations
try:
	from fse.exp.sif_variants_cy import sif_embeddings_6, \
		sif_embeddings_7, sif_embeddings_8
	CY_ROUTINES = 1
except ImportError as e:
	CY_ROUTINES = 0
	logger.warning("ImportError of Cython functions: %s", e)

# Simple in-place normalization
def normalize_text(sentences):
    for i, s in enumerate(sentences):
        sentences[i] = [sub("[^a-zA-Z]", "", w.lower()) for w in s]

if __name__ == "__main__":
	import pandas as pd
	import functools
	import psutil
	import timeit
	import argparse

	from gensim.models import Word2Vec
	from nltk.corpus import brown
	from collections import OrderedDict
	from datetime import datetime

	startTime = datetime.now()

	logging.basicConfig(
	    format='%(asctime)s : %(levelname)s : %(message)s',
	    level=logging.WARNING
	)
	logger.warning("running %s", " ".join(sys.argv))
	logger.warning("using cython routines %s", (CY_ROUTINES & CY_ROUTINES_TRAIN))

	parser = argparse.ArgumentParser()
	parser.add_argument("-len", help="Determine the length of the set to benchmark on", type=int, default=400)
	parser.add_argument("-runs", help="Determine the number of runs used to benchmark", type=int, default=1000)
	parser.add_argument("-size", help="Set the size of the embedding", type=int, default=100)
	parser.add_argument("-excel", help="Output results as excel file", type=int, default=0)
	args = parser.parse_args()

	# Prepare the brown corpus for the benchmark (~50k sentences)
	logger.warning("loading and preparing brown corpus")
	sentences = [s for s in brown.sents()]
	normalize_text(sentences)

	# Use a simple word2vec model for estimation
	# Training the model is not necessary for the comparision, the random wv.vectors are sufficient
	logger.warning("train word2vec model on corpus")
	model = Word2Vec(size=args.size, iter=1, workers=psutil.cpu_count(), sg=1, window=5, negative=5, min_count=5)
	model.build_vocab(sentences)

	# Precomputes sif weights for the final model
	se_model = Sentence2Vec(model, alpha=1e-3, components=0)

	# Precomputes the sif weights and sif weighted vectors for the benchmark of some python functions
	model.wv.sif = se_model._precompute_sif_weights(model.wv, alpha=1e-3)
	model.wv.sif_vectors = (model.wv.vectors * model.wv.sif[:, None]).astype(np.float32)

	# Precompute the word-indices list for the sentences (only for comparision)
	sentences_idx = [np.asarray([int(model.wv.vocab[w].index) for w in s if w in model.wv.vocab], dtype=np.intc) for s in sentences]

	# Use reduced size dataset
	data = sentences[:args.len]
	data_idx = sentences_idx[:args.len]
	results = OrderedDict()

	# The first verbose implementation is our reference implementation
	# All subequent computations must be allclose to the baseline
	baseline = None

	emb_dta = [(sif_embeddings, data),
				(sif_embeddings_1, data),
				(sif_embeddings_2, data),
				(sif_embeddings_3, data),
				(sif_embeddings_4, data),
				(sif_embeddings_5, data_idx)]
	if CY_ROUTINES:
		emb_dta = emb_dta + [
				(sif_embeddings_6, data_idx),
				(sif_embeddings_7, data_idx),
				(sif_embeddings_8, data_idx),
				(se_model.train, data)
				]
	# Note: We do not benchmark the "train" routine, as it containes estimate memory and logging, which would disturb the result.

	for i, tup in enumerate(emb_dta):
		func = tup[0]						# Function to evaluate
		dta = tup[1]						# Data to perform evaluation on
		n = args.runs if i is not 0 else 1	# Limit the first loop. Otherwise it takes ages.

		t = timeit.Timer(functools.partial(func, sentences=dta, model=model)) 
		time = np.min(t.repeat(number=n))
		logger.warning("compute embeddings with function: %s takes %2.6f sec", func.__name__, time / n)
		results[str(func.__name__)] = float(time) / n

		if i == 0:
			# Set the baseline up
			baseline = func(dta, model)
		else:
			# Test that all implementations are close to the baseline
			assert np.allclose(baseline, func(sentences=dta, model=model), atol=1e-6)

	
	# Compute result & store
	df = pd.DataFrame(results, columns=results.keys(), index=["Time(s)"]).T
	values = df["Time(s)"].values
	df["Gain"] = [1] + [values[i-1]/values[i] for i in range(1, len(values))]

	print("--- Results ---")
	print(df)

	now = datetime.now()
	date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
	
	if args.excel:
		p = pathlib.Path("excel")
		p.mkdir(exist_ok=True)
		df.to_excel("excel/results_"+date_time+".xlsx")

	logger.info("TOTAL RUNTIME: %s",str(now - startTime))