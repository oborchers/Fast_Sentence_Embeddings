import numpy as np
from re import sub
import logging
import os
import sys
import pathlib

from fse.models import Sentence2Vec
from fse.models.sentence2vec import CY_ROUTINES

from gensim.models.word2vec import Word2Vec, LineSentence

np.random.seed(42)
logger = logging.getLogger(__name__)

# Simple in-place normalization
def normalize_text(sentences):
    for i, s in enumerate(sentences):
        sentences[i] = [sub("[^a-zA-Z]", "", w.lower()) for w in s]

if __name__ == "__main__":
	import timeit
	import psutil
	import argparse

	from gensim.models import Word2Vec
	from nltk.corpus import brown
	
	from datetime import datetime

	startTime = datetime.now()

	logging.basicConfig(
	    format='%(asctime)s : %(levelname)s : %(message)s',
	    level=logging.INFO
	)
	logger.info("running %s", " ".join(sys.argv))
	logger.info("using cython routines %s", CY_ROUTINES)

    
	parser = argparse.ArgumentParser()

	# Training Parameters
	parser.add_argument("-train", help="Use text data from file TRAIN to train the model")
	parser.add_argument("-save", help="Set to True to save model", type=bool, default=True)
	parser.add_argument("-window", help="Set max skip length WINDOW between words (default: 5)", type=int, default=5)
	parser.add_argument("-size", help="Set size of word vectors (default: 100)", type=int, default=100)
	parser.add_argument("-sample", help="Set subsampling threshold (default: 1e-4)", type=float, default=1e-4)
	parser.add_argument("-negative", help="Number of negative samples (default: 5)",type=int, default=5)
	parser.add_argument("-threads", help="Use THREADS threads (default: 4)", type=int, default=4)
	parser.add_argument("-iter", help="Run more training iterations (default: 5)", type=int, default=5)
	parser.add_argument("-min_count", help="This will discard words that appear less than MIN_COUNT times (default: 5)", type=int, default=5)
	parser.add_argument("-alpha", help="Set SIF alpha for weighted sum to ALPHA (default: 1e-3)",type=float,default=1e-3)
	parser.add_argument("-pc", help="Set number of removed componented to PC (default: 1)",type=int,default=1)
	args = parser.parse_args()

	if args.train:
		sentences = LineSentence(args.train)
	else:
		logger.info("loading and preparing brown corpus")
		sentences = [s for s in brown.sents()]
		normalize_text(sentences)

	logger.info("train word2vec model on corpus")
	model = Word2Vec(
    	sentences, size=args.size, min_count=args.min_count, workers=args.threads,
    	window=args.window, sample=args.sample, sg=1, hs=0,
    	negative=args.negative, cbow_mean=0, iter=args.iter
    	)

	sif_model = Sentence2Vec(model, alpha=args.alpha, components=args.pc)
	sif_emb = sif_model.train(sentences)

	sif_model.normalize(sif_emb)

	if args.save:
		now = datetime.now()
		date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

		p = pathlib.Path("model_data")
		p.mkdir(exist_ok=True)

		out_model = "model_data/model_"+date_time
		model.save(out_model + '.model')

		out_emb = "model_data/sif_"+date_time
		np.save(out_emb, sif_emb)
		logger.info("saved %s", out_emb)
		
	logger.info("TOTAL RUNTIME: %s",str(now - startTime))