FROM python:3.6.0

ARG gensim==3.8.3

RUN pip install -U pip 

RUN pip install scipy \
    smart_open \
    scikit-learn \
    wordfreq \
    huggingface-hub \
    psutil

RUN pip install -U "gensim==$gensim" pytest coverage

ADD . /home
WORKDIR /home
RUN rm -rf build dist

RUN pip install -e .

CMD [ "pytest", "-vv" ]