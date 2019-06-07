
		 STS Benchmark: Main English dataset
			    
	    Semantic Textual Similarity 2012-2017 Dataset

		    http://ixa2.si.ehu.eus/stswiki
				   

STS Benchmark comprises a selection of the English datasets used in
the STS tasks organized by us in the context of SemEval between 2012
and 2017.

In order to provide a standard benchmark to compare among systems, we
organized it into train, development and test. The development part
can be used to develop and tune hyperparameters of the systems, and
the test part should be only used once for the final system.

The benchmark comprises 8628 sentence pairs. This is the breakdown
according to genres and train-dev-test splits:

                train  dev test total 
        -----------------------------
        news     3299  500  500  4299
        caption  2000  625  525  3250
        forum     450  375  254  1079
        -----------------------------
        total    5749 1500 1379  8628

For reference, this is the breakdown according to the original names
and task years of the datasets:

  genre     file           years   train  dev test
  ------------------------------------------------
  news      MSRpar         2012     1000  250  250
  news      headlines      2013-16  1999  250  250 
  news      deft-news      2014      300    0    0
  captions  MSRvid         2012     1000  250  250
  captions  images         2014-15  1000  250  250
  captions  track5.en-en   2017        0  125  125
  forum     deft-forum     2014      450    0    0
  forum     answers-forums 2015        0  375    0
  forum     answer-answer  2016        0    0  254
  
In addition to the standard benchmark, we also include other datasets
(see readme.txt in "companion" directory).


Introduction
------------

Given two sentences of text, s1 and s2, the systems need to compute
how similar s1 and s2 are, returning a similarity score between 0 and
5. The dataset comprises naturally occurring pairs of sentences drawn
from several domains and genres, annotated by crowdsourcing. See
papers by Agirre et al. (2012; 2013; 2014; 2015; 2016; 2017).

Format
------

Each file is encoded in utf-8 (a superset of ASCII), and has the
following tab separated fields:

  genre filename year score sentence1 sentence2

optionally there might be some license-related fields after sentence2.

NOTE: Given that some sentence pairs have been reused here and
elsewhere, systems should NOT use the following datasets to develop or
train their systems (see below for more details on datasets):

- Any of the datasets in Semeval STS competitions, including Semeval
  2014 task 1 (also known as SICK).
- The test part of MSR-Paraphrase (development and train are fine).
- The text of the videos in MSR-Video.


Evaluation script
-----------------

The official evaluation is the Pearson correlation coefficient. Given
an output file comprising the system scores (one per line) in a file
called sys.txt, you can use the evaluation script as follows:

$ perl correlation.pl sts-dev.txt sys.txt


Other
-----

Please check http://ixa2.si.ehu.eus/stswiki

We recommend that interested researchers join the (low traffic)
mailing list:

 http://groups.google.com/group/STS-semeval

Notse on datasets and licenses
------------------------------

If using this data in your research please cite (Agirre et al. 2017)
and the STS website: http://ixa2.si.ehu.eus/stswiki.

Please see LICENSE.txt
  

Organizers of tasks by year
---------------------------

2012 Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre

2013 Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre,
     WeiWei Guo

2014 Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab,
     Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau,
     Janyce Wiebe

2015 Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab,
     Aitor Gonzalez-Agirre, Weiwei Guo, Inigo Lopez-Gazpio, Montse
     Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, Janyce
     Wiebe

2016 Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor
     Gonzalez-Agirre, Rada Mihalcea, German Rigau, Janyce
     Wiebe

2017 Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia
     Specia


References
----------

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre. Task 6: A
   Pilot on Semantic Textual Similarity. Procceedings of Semeval 2012

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, WeiWei
   Guo. *SEM 2013 shared task: Semantic Textual
   Similarity. Procceedings of *SEM 2013

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab,
   Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau,
   Janyce Wiebe. Task 10: Multilingual Semantic Textual
   Similarity. Proceedings of SemEval 2014.

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab,
    Aitor Gonzalez-Agirre, Weiwei Guo, Inigo Lopez-Gazpio, Montse
    Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, Janyce
    Wiebe. Task 2: Semantic Textual Similarity, English, Spanish and
    Pilot on Interpretability. Proceedings of SemEval 2015.

Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor
    Gonzalez-Agirre, Rada Mihalcea, German Rigau, Janyce
    Wiebe. Semeval-2016 Task 1: Semantic Textual Similarity,
    Monolingual and Cross-Lingual Evaluation. Proceedings of SemEval
    2016.

Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia
    Specia. Semeval-2017 Task 1: Semantic Textual Similarity
    Multilingual and Crosslingual Focused Evaluation. Proceedings of
    SemEval 2017.

Clive Best, Erik van der Goot, Ken Blackler, Tefilo Garcia, and David
    Horby. 2005. Europe media monitor - system description. In EUR
    Report 22173-En, Ispra, Italy.

Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier.
    Collecting Image Annotations Using Amazon's Mechanical Turk.  In
    Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and
    Language Data with Amazon's Mechanical Turk.




