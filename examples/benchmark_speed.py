import numpy as np
from re import sub
import logging
import sys

np.random.seed(42)
logger = logging.getLogger(__name__)

from fse import SIF
from fse.sif import CY_ROUTINES as CY_ROUTINES_TRAIN

from fse.sif_variants import sif_embeddings, \
	sif_embeddings_1, sif_embeddings_2, sif_embeddings_3, \
	sif_embeddings_4, sif_embeddings_5

try:
	# Import cython functions  
	from fse.sif_variants_cy import sif_embeddings_6, \
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

	from gensim.fse import Word2Vec
	from nltk.corpus import brown
	from collections import OrderedDict
	from datetime import datetime

	startTime = datetime.now()

	logging.basicConfig(
	    format='%(asctime)s : %(levelname)s : %(message)s',
	    level=logging.INFO
	)
	logger.info("running %s", " ".join(sys.argv))
	logger.info("using cython routines %s", (CY_ROUTINES & CY_ROUTINES_TRAIN))

	parser = argparse.ArgumentParser()
	parser.add_argument("-len", help="Determine the length of the set to benchmark on", type=int, default=400)
	parser.add_argument("-runs", help="Determine the number of runs used to benchmark", type=int, default=1000)
	parser.add_argument("-size", help="Set the size of the embedding", type=int, default=100)
	parser.add_argument("-train", help="Actually train the embedding", type=int, default=0)
	parser.add_argument("-excel", help="Output results as excel file", type=int, default=0)
	args = parser.parse_args()

	logger.info("loading and preparing brown corpus")
	sentences = [s for s in brown.sents()]
	normalize_text(sentences)

	logger.info("train word2vec model on corpus")
	model = Word2Vec(size=args.size, iter=1, workers=psutil.cpu_count(), sg=1, window=5, negative=5, min_count=5)
	model.build_vocab(sentences)
	if args.train:
		model.train(sentences, epochs=model.epochs, total_examples=model.corpus_count)

	# Precomputes sif weights and vectors
	sif_model = SIF(model, alpha=1e-3, components=0)
	sif_model.precompute_sif_vectors(model.wv, 1e-3)
	model.wv.sif_vectors = sif_model.sif_vectors
	model.wv.sif = sif_model.sif
	
	sentences_idx = [np.asarray([int(model.wv.vocab[w].index) for w in s if w in model.wv.vocab], dtype=np.intc) for s in sentences]

	data = sentences[:args.len]
	data_idx = sentences_idx[:args.len]
	results = OrderedDict()		# Save results

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
				(sif_model.train, data),
				]

	for i, tup in enumerate(emb_dta):
		func = tup[0]					# Function to evaluate
		dta = tup[1]					# Data to perform evaluation on
		n = args.runs if i is not 0 else 1	# Limit the first loop. Otherwise it takes ages.

		t = timeit.Timer(functools.partial(func, dta, model)) 
		time = np.min(t.repeat(number=n))
		logger.info("compute embeddings with function: %s takes %2.6f sec", func.__name__, time / n)
		results[str(func.__name__)] = float(time) / n
		if i == 0:
			baseline = func(dta, model)
		else:
			assert np.allclose(baseline, func(dta, model), atol=1e-6)

	
	df = pd.DataFrame(results, columns=results.keys(), index=["Time(s)"]).T
	values = df["Time(s)"].values
	df["Gain"] = [1] + [values[i-1]/values[i] for i in range(1, len(values))]
	print(df)

	now = datetime.now()
	date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

	if args.excel:
		df.to_excel("excel/results_"+date_time+".xlsx")

	logger.info("TOTAL RUNTIME: %s",str(now - startTime))