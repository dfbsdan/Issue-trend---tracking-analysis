# # TASK 1
# Based on:
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24#:~:text=Topic%20modeling%20is%20a%20type,document%20to%20a%20particular%20topic.
# https://dzone.com/articles/topic-modelling-techniques-and-ai-models

import json
import glob
import time
import zipfile
from matplotlib import pyplot as plt
import pandas as pd
from psutil import cpu_count
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer, pos_tag_sents
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaMulticore, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from tqdm import notebook

# PREPROCESSING
def preprocess_sentence(sent: list, lemma: WordNetLemmatizer, stop_words: set):
  # remove stopwords
  sent = [word for word in sent if not word in stop_words]
  # normalize
  sent = [lemma.lemmatize(word, pos="v") for word in sent]
  sent = [lemma.lemmatize(word, pos="n") for word in sent]
  return sent

def tokenize_body(body: list):
  def tokenize_sentence(sent: str):
    # divide sentence into words, remove punctuations and turn to lowercase
    return [word.lower() for word in word_tokenize(sent) if word.isalpha()]

  return [tokenize_sentence(sent) for sent in sent_tokenize(body.replace('.', '. '))]

def preprocess_data(path: str):
  with zipfile.ZipFile(path, 'r') as z_file:
    z_file.extractall()
  lemma = WordNetLemmatizer()
  stop_words = set(stopwords.words("english"))
  data = list()
  for f_name in glob.glob('./data/koreaherald_*.json'):
    with open(f_name, 'r') as f:
      d: dict = json.load(f)
      titles: dict = d['title']
      times: dict = d[' time']
      bodies: dict = d[' body']
      secs: dict = d[' section']
      raw_sents = {idx:tokenize_body(body) for idx, body in bodies.items()}
      norm_sents = {idx:[preprocess_sentence(sent, lemma, stop_words) for sent in sents] for idx, sents in raw_sents.items()}
      pos_tagged = {idx:pos_tag_sents(sents) for idx, sents in raw_sents.items()}
      data += [(title, times[idx], secs[idx], pos_tagged[idx], raw_sents[idx], norm_sents[idx]) for idx, title in titles.items() if len(bodies[idx]) > 0]
  data = pd.DataFrame(data, columns=['title', 'time', 'section', 'tagged_sentences', 'raw_sentences', 'norm_sentences'])
  return data

# DATA SETUP
def get_vocabulary(data: pd.DataFrame, min_docs: int, max_docs: float):
  vocab = Dictionary(data['words'])
  vocab.filter_extremes(no_below=min_docs, no_above=max_docs)
  data['bow'] = [vocab.doc2bow(text) for text in data['words']]
  return vocab

def get_data(data_path: str, load: bool, save_path):
  if load:
    data: pd.DataFrame = pd.read_json(data_path) 
  else:
    data = preprocess_data(data_path)
  if isinstance(save_path, str):
    data.to_json(save_path)
  data['time'] = pd.to_datetime(data['time'])
  data['words'] = data['norm_sentences'].apply(lambda text: [word for sent in text for word in sent])
  return data

# LDA
def train_lda(data: pd.DataFrame, vocab: Dictionary, 
    min_k: int, max_k: int, k_step: int, 
    passes:int, workers: int, timeout: int, 
    plot: bool, verbose: bool, store_vis_path) -> LdaMulticore:

  assert max_k >= min_k and min_k > 0 and k_step > 0 and timeout >= 0
  timeout *= 60 # timeout given in minutes
  corpus = TfidfModel(list(data['bow']))[data['bow']]
  ks = list(range(min_k, max_k + 1, k_step))
  accuracies = list()
  t0 = time.time()
  for i, k in enumerate(ks):
    t_start = time.time()
    model = LdaMulticore(corpus, num_topics=k, id2word=vocab, passes=passes, workers=workers)
    coherence = CoherenceModel(model=model, texts=data['words'], dictionary=vocab, coherence='c_v', processes=workers).get_coherence()
    perplexity = model.log_perplexity(data['bow']).sum()
    t_end = time.time()
    if verbose:
      print(f'Finished training for: {k} issues ({i+1}/{len(ks)}) - {int(t_end - t_start)}secs\n    Coherence: {coherence}\n    Perplexity: {perplexity}')
    accuracies.append((k, coherence, perplexity))
    if isinstance(store_vis_path, str):
      try:
        pyLDAvis.save_html(gensimvis.prepare(model, corpus, vocab), f'{store_vis_path}/{k}_{passes}.html')
      except Exception as e:
        print(f'ERROR: Could not store visualization for {k} issues: {e}')
    if timeout > 0 and (time.time() - t0) >= timeout:
      print('\n-----------------------------------TIMEOUT-----------------------------------\n')
      break
  # get the best model
  if len(accuracies) > 1: # more than one model trained
    coherences = sorted([(k, coh) for k, coh, _ in accuracies], key=lambda tup: tup[1], reverse=True)
    perplexities = sorted([(k, perp) for k, _, perp in accuracies], key=lambda tup: tup[1])
    k_stats = {k:[0, coh, perp] for k, coh, perp in accuracies}
    for i, ((k1, _), (k2, _)) in enumerate(zip(coherences, perplexities)):
      k_stats[k1][0] += i
      k_stats[k2][0] += i
    k_stats = sorted([(k, score, coh, perp) for k, (score, coh, perp) in k_stats.items()], key=lambda tup: tup[1])
    best_k, _, coherence, perplexity = k_stats[0]
    best_model = LdaMulticore(corpus, num_topics=best_k, id2word=vocab, passes=passes, workers=workers)
  else: # only one model trained
    assert len(accuracies) == 1
    best_model = model
    best_k, coherence, perplexity = accuracies[0]
  if verbose:
    print(f'\nBEST MODEL: Issues: {best_k}\n    Coherence: {coherence}\n    Perplexity: {perplexity}')
    print('Sample Issues:')
    for issue_idx, words in best_model.print_topics(num_topics=10, num_words=10):
      print(f'    Issue: {issue_idx}\n        {words}')
  if plot:
    t = int(time.time())
    plt.plot(ks[:len(accuracies)], [coh for _, coh, _ in accuracies])
    plt.xlabel("Number of issues (k)")
    plt.ylabel("Coherence")
    plt.savefig(f'./coherence_{t}.png')
    plt.clf()
    plt.plot(ks[:len(accuracies)], [perp for _, _, perp in accuracies])
    plt.xlabel("Number of issues (k)")
    plt.ylabel("Perplexity")
    plt.savefig(f'./perplexity_{t}.png')
  return best_model

def get_issue_name(issue_idx: int, model: LdaMulticore, issue_words: int):
  assert issue_idx >= 0 and issue_words > 0
  return "-".join([word for word, _ in model.show_topic(issue_idx, topn=issue_words)])

def issues_per_doc(data: pd.DataFrame, model: LdaMulticore) -> pd.Series:
  return data['bow'].apply(lambda bow: sorted([(issue_idx,issue_prob) for issue_idx, issue_prob in model[bow]], key=lambda tup: tup[1], reverse=True))

# returns an ordered list of pairs: (issue_idx, issue_score)
def rank_issues(data: pd.DataFrame, doc_to_issues: pd.Series):
  doc_cnt = len(data)
  assert doc_cnt == len(doc_to_issues)
  issue_to_score = dict()
  for issue_list in doc_to_issues:
    best_issue_idx, _ = issue_list[0]
    if best_issue_idx in issue_to_score:
      issue_to_score[best_issue_idx] += 1
    else:
      issue_to_score[best_issue_idx] = 1
  return sorted(list(issue_to_score.items()), key=lambda tup: tup[1], reverse=True)

load = True
min_docs = 2 # min amount of docs a word needs to appear on
max_docs = 0.7 # max ratio of documents a word is allowed to appear on
data_path = './data/lda_data.json'# if load else './dataset_korea_herald.zip'
save_path = None# if load else './data/lda_data.json'
min_k = 10
max_k = 200
k_step = 10
issue_words = 5
use_best_k = True
passes = 5
workers = cpu_count(logical=False) - 1
timeout = 40 # mins
plot = False
verbose = False
store_vis_path = None#'./visualizations'

def get_year_issues(year: str, data: pd.DataFrame, vocab: Dictionary, return_mdata: bool):
  best_k = {'2017': 50, '2016': 30, '2015': 80}
  params = {
    'data': data,
    'vocab': vocab,
    'min_k': best_k[year] if use_best_k else min_k,
    'max_k': best_k[year] if use_best_k else max_k,
    'k_step': k_step,
    'passes': passes,
    'workers': workers,
    'timeout': timeout,
    'plot': plot,
    'verbose': verbose,
    'store_vis_path': store_vis_path,
  }
  model = train_lda(**params)
  doc_to_issues = issues_per_doc(data, model)
  year_issues = rank_issues(data, doc_to_issues)[:10]
  year_issues = [(issue_idx, get_issue_name(issue_idx, model, issue_words), issue_score) for issue_idx, issue_score in year_issues]
  if return_mdata:
    return {
      'issues': year_issues,
      'model': model,
      'doc_to_issues': doc_to_issues,
      'data': data,
    }
  return year_issues

def get_issues(data: pd.DataFrame, return_mdata: bool):
  data_2017: pd.DataFrame = data[(data['time'] >= pd.to_datetime('20170101', format='%Y%m%d'))].copy()
  vocab_2017 = get_vocabulary(data_2017, min_docs, max_docs)
  data_2016: pd.DataFrame = data[(data['time'] >= pd.to_datetime('20160101', format='%Y%m%d')) & (data['time'] <= pd.to_datetime('20161231', format='%Y%m%d'))].copy()
  vocab_2016 = get_vocabulary(data_2016, min_docs, max_docs)
  data_2015: pd.DataFrame = data[(data['time'] >= pd.to_datetime('20150101', format='%Y%m%d')) & (data['time'] <= pd.to_datetime('20151231', format='%Y%m%d'))].copy()
  vocab_2015 = get_vocabulary(data_2015, min_docs, max_docs)
  t_2017 = ('2017', data_2017, vocab_2017)
  t_2016 = ('2016', data_2016, vocab_2016)
  t_2015 = ('2015', data_2015, vocab_2015)
  return [get_year_issues(*y_tup, return_mdata) for y_tup in (t_2017, t_2016, t_2015)]

def task1_lda():
  # print the best issues for each year
  def print_issues(year: str, issues: list):
    print(f'    {year} : ' + ', '.join([issue_name for _, issue_name, _ in issues]))

  issues_2017, issues_2016, issues_2015 = get_issues(get_data(data_path, load, save_path), False)
  print_issues('2015', issues_2015)
  print_issues('2016', issues_2016)
  print_issues('2017', issues_2017)

task1_lda()

# # TASK 2
# Based on:
# https://towardsdatascience.com/natural-language-processing-event-extraction-f20d634661d3

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances_argmin_min
import spacy

similarity_threshold = 0.561
equality_threshold = 0.1
relevant_pos = {
  'NN', 'NNS', 'NNP', 'NNPS', # nouns
  'RB', 'RBR', 'RBS', # adverbs
  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', # verbs
  'JJ', 'JJR', 'JJS', #adjectives
}
grid_search_eps = False

def print_issue_events(issue_name: str, events: list, related_events: list):
  def __print_events(events: list):
    event_names = list()
    event_specs = list()
    for event in events:
      event_names.append(event['name'])
      event_specs.append(f"Event: {event['name']}\n\n   -   Person: {', '.join(event['people'])}\n   -   Organization: {', '.join(event['orgs'])}\n   -   Place: {', '.join(event['places'])}\n")    
    print(f"{' -> '.join(event_names)}\n[ Detailed Information (per event) ]")
    for e_specs in event_specs:
      print(e_specs)

  # task 2.1
  print(f'[ Issue ]\n{issue_name}\n[ On-Issue Events ]')
  __print_events(events)
  # task 2.2
  print(f'[ Issue ]\n{issue_name}\n[ Related-Issue Events ]')
  __print_events(related_events)

def event_similarity(e1: dict, e2: dict):
  v1 = e1['emb']
  v2 = e2['emb']
  return 1 - dot(v1, v2)/(norm(v1)*norm(v2))

def remove_equal_events(raw_events: list):
  def __merge_events(e1: dict, e2: dict):
    e1['people'].update(e2['people'])
    e1['orgs'].update(e2['orgs'])
    e1['places'].update(e2['places'])
    e1['emb'] = (e1['emb'] + e2['emb']) / 2
    return e1

  if verbose:
    print('Removing equal events...')
  events = []
  while len(raw_events) > 0:
    e1: dict = raw_events.pop(0)
    equal_events = {i for i, e2 in enumerate(raw_events) if event_similarity(e1, e2) < equality_threshold}
    for i in equal_events:
      e1 = __merge_events(e1, raw_events[i])
    events.append(e1)
    raw_events = [e for i, e in enumerate(raw_events) if (not i in equal_events)]
  return events

def get_related_events(events1: list, events2: pd.Series):
  def __similar_events(events1: list, e2: dict):
    for e1 in events1:
      if event_similarity(e1, e2) < similarity_threshold:
        return True
    return False

  if verbose:
    print(f'Getting related events... ({time.time()})')
  events2 = events2[events2.apply(lambda e2: __similar_events(events1, e2))]
  return remove_equal_events(list(events2))

def cluster_text(raw_text: list, emb_model, dbscan: DBSCAN):
  # filter relevant sentences (each sentence must have at least two relevant pos tags)
  text = []
  for sent in raw_text:
    has_relev_pos = False
    for _, pos in sent:
      if pos in relevant_pos:
        if has_relev_pos:
          text.append(sent)
          break
        else:
          has_relev_pos = True
  if len(text) == 0: # no relevant sentences found
    return []
  sent_embeddings: np.ndarray = emb_model.encode([[word for word, pos in sent if pos in relevant_pos] for sent in text])
  assert len(sent_embeddings) == len(text)
  clusters = dbscan.fit(sent_embeddings).labels_
  assert len(clusters) == len(sent_embeddings)
  cluster_to_sents = dict()
  for cluster, embedding, sent in zip(clusters, sent_embeddings, text):
    if cluster >= 0:
      if cluster in cluster_to_sents:
        cluster_to_sents[cluster].append((embedding, sent))
      else:
        cluster_to_sents[cluster] = [(embedding, sent)]
  return list(cluster_to_sents.values())

def find_events(raw_text: list, time, emb_model, dbscan: DBSCAN, entity_recog): # must be ordered by time
  def __extract_entities(sent: list, entity_recog):
    assert len(sent) > 0
    orgs = set()
    places = set()
    people = set()
    for entity in entity_recog(' '.join([word for word, _ in sent])).ents:
      name = entity.text
      label = entity.label_
      if label == 'ORG':
        orgs.add(name)
      elif label == 'PERSON':
        people.add(name)
      elif label in {'LOC', 'GPE'}:
        places.add(name)
    return orgs, people, places

  def __get_event(cluster: list, entity_recog):
    assert len(cluster) > 0
    embeddings = np.array([emb for emb, _ in cluster])
    centroid = embeddings.sum(axis=0) / len(cluster)
    entities = [__extract_entities(sent, entity_recog) for _, sent in cluster]
    _, sent = cluster[pairwise_distances_argmin_min([centroid], embeddings)[0][0]] # sentence closest to centroid
    return {
      'name': ' '.join([word for word, pos in sent if pos in relevant_pos]),
      'people': set.union(*[s_people for _, s_people, _ in entities]),
      'orgs': set.union(*[s_orgs for s_orgs, _, _ in entities]),
      'places': set.union(*[s_places for _, _, s_places in entities]),
      'emb': centroid,
    }
  
  clusters = cluster_text(raw_text, emb_model, dbscan)
  return [__get_event(cluster, entity_recog) for cluster in clusters]

def get_issue_events(issue_idx: int, doc_to_issues: pd.Series, data: pd.DataFrame, emb_model, dbscan: DBSCAN, entity_recog):
  assert issue_idx >= 0
  selected_docs = [doc_idx for doc_idx, issues in enumerate(doc_to_issues) if issues[0][0] == issue_idx]
  assert len(selected_docs) > 0
  events = list()
  selected_docs: pd.DataFrame = data[['time', 'tagged_sentences']].iloc[selected_docs]
  selected_docs: pd.Series = selected_docs.apply(lambda doc: find_events(doc['tagged_sentences'], doc['time'], emb_model, dbscan, entity_recog), axis=1)
  for doc_events in selected_docs:
    if len(doc_events) > 0:
      events += doc_events
  return remove_equal_events(events)

def task2_lda():
  data_2017, data_2016, data_2015 = get_issues(get_data(data_path, load, save_path), True)
  issue_to_events = dict()
  emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
  dbscan = DBSCAN(eps=similarity_threshold, min_samples=2, metric='cosine')
  entity_recog = spacy.load("en_core_web_sm")
  for year, data_dict in (('2017', data_2017), ('2016', data_2016), ('2015', data_2015)):
    for issue_idx, issue_name, _ in data_dict['issues']:
      if verbose:
        print(f'Getting events for issue: {issue_idx} in year: {year}')
      issue_events = get_issue_events(issue_idx, data_dict['doc_to_issues'], data_dict['data'], emb_model, dbscan, entity_recog)
      if len(issue_events) > 0:
        issue_key = (year, issue_idx, issue_name)
        assert not issue_key in issue_to_events
        issue_to_events[issue_key] = issue_events
  issue_to_events = sorted(list(issue_to_events.items()), key=lambda tup: len(tup[1]), reverse=True)
  selected_issues = issue_to_events[:2]
  issue_to_events = pd.Series([event for _, events in issue_to_events[2:] for event in events])
  # get and print the issue events
  for (_, issue_idx, issue_name), issue_events in selected_issues:
    print_issue_events(issue_name, issue_events, get_related_events(issue_events, issue_to_events))

def test_epsilon(eps_min: float = 0.001, eps_max: float = 1.0, eps_step: float = 0.02):
  assert eps_min > 0 and eps_max > eps_min and eps_step > 0
  data_2017, data_2016, data_2015 = get_issues(get_data(data_path, load, save_path), True)
  emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
  tests = np.arange(eps_min, eps_max, eps_step)
  clusters_tot = [None] * len(tests)
  max_tot = [-1, -1]
  clusters_avg = [None] * len(tests)
  max_avg = [-1, -1]
  for i, similarity_threshold in notebook.tqdm(enumerate(tests), total=len(tests)):
    cluster_cnt = 0
    doc_cnt = 0
    dbscan = DBSCAN(eps=similarity_threshold, min_samples=2, metric='cosine')
    for data_dict in (data_2017, data_2016, data_2015):
      tagged_sents: pd.Series = data_dict['data']['tagged_sentences']
      issue_to_docs = dict()
      for doc_idx, issues in enumerate(data_dict['doc_to_issues']):
        issue_idx = issues[0][0]
        if issue_idx in issue_to_docs:
          issue_to_docs[issue_idx].append(doc_idx)
        else:
          issue_to_docs[issue_idx] = [doc_idx]
      for issue_idx, _, _ in data_dict['issues']:        
        selected_docs: pd.Series = tagged_sents.iloc[issue_to_docs[issue_idx]]
        doc_cnt += len(selected_docs)
        for raw_text in selected_docs:
          cluster_cnt += len(cluster_text(raw_text, emb_model, dbscan))
    clusters_tot[i] = cluster_cnt
    if cluster_cnt > max_tot[0]:
      max_tot[0] = cluster_cnt
      max_tot[1] = similarity_threshold
    avg_clusters = cluster_cnt / doc_cnt
    clusters_avg[i] = avg_clusters
    if avg_clusters > max_avg[0]:
      max_avg[0] = avg_clusters
      max_avg[1] = similarity_threshold
    print(f'Eps: {similarity_threshold}, tot clusters: {cluster_cnt}, avg clusters: {avg_clusters}')
  print(f'Max total clusters: {max_tot[0]} for eps: {max_tot[1]}')
  print(f'Max average clusters: {max_avg[0]} for eps: {max_avg[1]}')
  plt.plot(tests, clusters_tot, label="Total clusters")
  plt.plot(tests, clusters_avg, label="Average clusters (per doc)")
  plt.xlabel("Epsilon")
  plt.ylabel("Number of clusters")
  plt.legend()
  plt.show()

task2_lda()
if grid_search_eps:
  test_epsilon()