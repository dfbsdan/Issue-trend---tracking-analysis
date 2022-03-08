#!/bin/bash
python3 -m spacy download en 
python3 -m spacy download en_core_web_sm

echo 'STARTING: TASK 1-2 with CLUSTERING'
python3 task1_2-clustering.py

echo 'STARTING: TASK 1-2 with LDA and CLUSTERING'
python3 task1_2-lda_&_clustering.py