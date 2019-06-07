import numpy as np

# Define data types for use in cython
REAL = np.float32 
INT = np.intc

def sif_embeddings(sentences, model, alpha=1e-3):
    """Compute the SIF embeddings for a list of sentences

    Parameters
    ----------
    sentences : list
        The sentences to compute the embeddings for
    model : `~gensim.models.base_any2vec.BaseAny2VecModel`
        A gensim model that contains the word vectors and the vocabulary
    alpha : float, optional
        Parameter which is used to weigh each individual word based on its probability p(w).

    Returns
    -------
    numpy.ndarray 
        SIF sentence embedding matrix of dim len(sentences) * dimension
    """
    
    vlookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    size = model.vector_size  # Embedding size
    
    Z = 0
    for k in vlookup:
        Z += vlookup[k].count # Compute the normalization constant Z
    
    output = []
    
    # Iterate all sentences
    for s in sentences:
        count = 0
        v = np.zeros(size, dtype=REAL) # Summary vector
        # Iterare all words
        for w in s:
            # A word must be present in the vocabulary
            if w in vlookup:
                for i in range(size):
                    v[i] += ( alpha / (alpha + (vlookup[w].count / Z))) * vectors[w][i]
                count += 1 
                
        if count > 0:
            for i in range(size):
                v[i] *= 1/count
        output.append(v)
    return np.vstack(output).astype(REAL)


def sif_embeddings_1(sentences, model, alpha=1e-3):
	""" Removes the unecessary loop in the vector summation
	"""
	vlookup = model.wv.vocab
	vectors = model.wv
	size = model.vector_size

	Z = 0
	for k in vlookup:
	    Z += vlookup[k].count

	output = []
	for s in sentences:
	    count = 0
	    v = np.zeros(size, dtype=REAL)
	    for w in s:
	        if w in vlookup:
	        	# The loop over the the vector dimensions is completely unecessary and extremely slow
	            v += ( alpha / (alpha + (vlookup[w].count / Z))) * vectors[w]
	            count += 1
	    if count > 0:
	        v *= 1/count
	    output.append(v)
	return np.vstack(output).astype(REAL)


def sif_embeddings_2(sentences, model, alpha=1e-3):
	""" Uses the precomputed SIF weights via lookup
	"""
	vlookup = model.wv.vocab
	vectors = model.wv
	size = model.vector_size

	output = []
	for s in sentences:
	    count = 0
	    v = np.zeros(size, dtype=REAL)
	    for w in s:
	        if w in vlookup:
	            v +=  vectors.sif[vlookup[w].index]*vectors[w]
	            count += 1
	    if count > 0:
	        v *= 1/count
	    output.append(v)
	return np.vstack(output).astype(REAL)


def sif_embeddings_3(sentences, model, alpha=1e-3):
	""" Precomputes the indices of the sentences and uses the numpy indexing to directly multiply and sum the vectors
	"""
	vlookup = model.wv.vocab
	vectors = model.wv

	output = []
	for s in sentences:
	    
	    idx = [vlookup[w].index for w in s if w in vlookup]
	    
	    v = np.sum(vectors.vectors[idx] * vectors.sif[idx][:, None], axis=0)
	    if len(idx) > 0:
	        v *= 1/len(idx)
	    output.append(v)
	return np.vstack(output).astype(REAL)


def sif_embeddings_4(sentences, model):
	""" Precomputes the sif_vectors in a separate matrix
	"""
	vlookup = model.wv.vocab
	vectors = model.wv.sif_vectors

	output = []
	for s in sentences:
	    idx = [vlookup[w].index for w in s if w in vlookup]
	    v = np.sum(vectors[idx], axis=0)
	    if len(idx) > 0:
	        v *= 1/len(idx)
	    output.append(v)
	return np.vstack(output).astype(REAL)


def sif_embeddings_5(sentences, model):
	""" Uses a pre-computed list of indices and skips the use of strings alltogether
	"""
	vectors = model.wv.sif_vectors
	output = np.zeros(shape=(len(sentences), model.vector_size), dtype=REAL)

	for i,s in enumerate(sentences):
		output[i] = np.sum(vectors[s], axis=0) * ( (1/len(s)) if len(s)>0 else 1)
	return output.astype(REAL)