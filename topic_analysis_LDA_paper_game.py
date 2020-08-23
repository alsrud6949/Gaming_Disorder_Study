# -*- coding: utf-8 -*-
"""
Topic Analysis - LDA model

"""


# Morphology Analysis
'''
import os
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Read file
os.chdir('C:\\Users\\~') #designate file's location
df = pd.read_excel("paper_game.xlsx") #file including papers about game
df = df.dropna(subset = ['abstract'])

# Convert to list
data = df.abstract.values.tolist() 

# Abreviation
# Dictionary of list of abreviation wrotten contraily
tmp_dic = {"Flying Ad-Hoc Network " :  "FANET", "interference-aware online channel preserving based concurrent best response " :  "IOCPCBR", "Motion-onset visually evoked potentials " :  "mVEPs", "Line-of-sight " :  "LOS", "coalitional game " :  "CG", "analytic hierarchy process " :  "AHP", "3rd Generation Partnership Project " :  "3GPP", "Narrow-band Internet of Things " :  "NB-IoT", "quality of service " :  "QoS", "physical mobile network operator " :  "PMNO", "virtual mobile network operators " :  "VMNOs", "objective structured clinical exams " :  "OSCEs", "Standardized patients " :  "SPs", "computer-generated imagery " :  "CGI ", "fuzzy-set qualitative comparative analysis " :  "fsQCA", "content providers " :  "CPs", "Mobile Ad-hoc Network " :  "MANET", "quality-of-service " :  "QoS", "QoS-aware game theory based power control " :  "QoS-GTPC", "at-risk and problem gambling " :  "ARPG", "grade point average " :  "GPA", "Internet Gaming Disorder Scale" : "IGDS", "Parental version of the Internet Gaming Disorder Scale" : "PIGDS", "Korean IGD-20 " :  "K-IGD-20", "active video games " :  "AVGs", "randomized controlled trials " :  "RCTs", "heterogeneous networks " :  "HetNets", "enhanced intercell interference coordination " :  "eICIC", "macrocell base station " :  "MBS", "resource allocation " :  "RA", "small cell base station " :  "SBS", "almost blank subframe " :  "ABS", "macrocell users " :  "MUs", "Stackelberg equilibrium " :  "SE", "hybrid data stream " :  "FGCH", "clustering center fast determination algorithm " :  "CCFD", "incremental redundancy-hybrid automatic repeat request " :  "IR-HARQ", "HARQ Markov model " :  "HARQ-MM", "Human Machine Interfaces " :  "HMIs", "triboelectric nanogenerators " :  "TENG", "time-of-flight " :  "ToF", "multiple-choice questions " :  "MCQs", "ensemble empirical mode decomposition " :  "EEMD", "online to offline " :  "O2O", "closed loop supply chain " :  "CLSC", "Episodic recent thinking " :  "ERT", "Episodic future thinking " :  "EFT ", "Delay discounting " :  "DD", "standardized episodic thinking " :  "SET", "Composite Infrared Spectrometer " :  "CIRS", "mobile ad hoc networks " :  "MANETs", "game experience " :  "GE", "Online Flow Questionnaire " :  "OFQ", "Major depressive episodes " :  "MDEs", "social welfare " :  "SW", "in-app purchases " :  "IAPs", "user-generated content " :  "UGC", "fifth-generation " :  "5G",  "further education " :  "FE", "higher education " :  "HE", " DSM-5 " : " DIAGNOSTIC AND STATISTICAL MANUAL OF MENTAL DISORDERS " , " PROBLEMATIC INTERNET USE " : " PIU " , " INTERNET ADDICTION " : " IA " , " AUTISM SPECTRUM DISORDER " : " ASD " , " INTERNET GAMING DISORDER " : " IGD " , " PHYSICAL ACTIVITY " : " PA " , " OBSESSIVE-COMPULSIVE " : " OCPD " , " PARKINSONS DISEASE " : " PD  " , " SLEEP-RELATED EATING DISORDERSRED " : " SRED " , " JUVENILE SYSTEMIC LUPUS " : " JSL  " , " SMARTPHONE USE DISORDER " : " SUD  " , " OPTIMIZED VERSION OF CROW SEARCH ALGORITHM " : " OCSA  " , " DEVELOPMENTAL COORDINATION DISORDER " : " DCD  " , " JUVENILE SYSTEMIC LUPUS ERYTHEMATOSUS " : " JSLE  " , " SMALL-VESSEL DISEASE " : " SVD  " , " SMALL VESSEL AND LACUNAR " : " SVE-LA  " , " PHARMACY-BASED COST GROUP " : " PCG  " , " REPETITIVE BEHAVIOUR QUESTIONNAIRE " : " RBQ  " , " CEREBRAL PALSY " : " CP " , " CAFFEINE USE DISORDER " : " CUD " , " CAFFEINE USE DISORDER QUESTIONNAIRE " : " CUDQ " , " ITEM RESPONSE THEORY " : " IRT " , " MILD TRAUMATIC BRAIN INJURY " : " MTBI " , " WORLD HEALTH ORGANIZATION " : " WHO " , " POLYCHLORINATED BIPHENYLS " : " PCBS " , " SUDDEN UNEXPLAINED DEATHS " : " SUD " , " SEROTONIN TRANSPORTER GENE " : " SERT " , " ADVERSE DRUG REACTIONS " : " ADRS " , " TRAUMATIC BRAIN INJURY " : " TBI " , " DIRECT CURRENT STIMULATION " : " TDCS " , " PEOPLE LIVING WITH HIV " : " PLWH " , " SPEED OF PROCESSING COGNITIVE REMEDIATION THERAPY " : " SOP-CRT " , " HEPATIC ENCEPHALOPATHY " : " HE " , " JAPANESE SLEEP QUESTIONNAIRE FOR ELEMENTARY SCHOOLERS " : " JSQ-ES " , " QUANTITATIVE ELECTROENCEPHALOGRAPHY " : " QEEG " , " NEUROGENIC DETRUSOR OVERACTIVITY " : " NDO " , " BOTULINUM TOXIN A " : " BONTA " , " DETRUSOR OVERACTIVITY " : " DO " , " POST-CONCUSSION SYNDROME " : " PCS " , " NIH " : " NATIONAL INSTITUTES OF HEALTH " , " TYPE 1 DIABETES " : " T1D " , " OSTEOCHONDRITIS DISSECANS " : " OCD " , " LEISURE-TIME SEDENTARY BEHAVIORS " : " LTSBS " , " HEPATITIS C VIRUS " : " HCV " , " PRIMARY EFFUSION LYMPHOMA " : " PEL " , " COGNITIVE ERRORS QUESTIONNAIRE " : " CEQ " , " COGNITIVE ERRORS QUESTIONNAIRE-REVISED " : " CEQ-R " , " BENIGN PAROXYSMAL VERTIGO OF CHILDHOOD " : " BPV " , " ARTIFICIAL INTELLIGENCE " : " AI " , " SELF-INJURIOUS BEHAVIOUR " : " SIB " , " INTEGRATED MANAGEMENT OF CHILDHOOD ILLNESS " : " IMCI " , " DEXAMETHASONE SUPPRESSION TEST " : " DST " , " CLONIDINE STIMULATION TEST " : " CST " , " SPINA BIFIDA MENINGOMYELOCELE " : " SBM  " , " EVENT-RELATED BRAIN POTENTIAL " : " ERP " , " INDUSTRIAL INFORMATION INTEGRATION ENGINEERING " : " IIIE " , " WOMEN'S ARMY CORPS " : " WAC " , " MONOAMINE OXIDASE " : " MAO " , " REACTION TIME " : " RT " , " PERCEPTUAL MAZE TEST " : " PMT " , " ECOLOGICAL TASK ANALYSIS " : " ETA " , " DISTRIBUTED AI " : " DAI " , " IMMUNOGLOBULINS " : " IG " , " IMMUNOGLOBULIN G " : " IGG " , " IMMUNOGLOBULIN M " : " IGM " , " INTERNET GAMING DISORDER SCALE " : " IGDS " , " VIRTUAL SELF-DISCREPANCY " : " VSD " , " MASSIVELY MULTIPLAYER ONLINE ROLE-PLAYING GAMES " : " MMORPGS " , " PATHOLOGIC VIDEO GAME USE " : " PVGU " , " GAME TRANSFER PHENOMENA " : " GTP " , " METHYLPHENIDATE " : " MPH " , " ATTENTION-DEFICIT/HYPERACTIVITY DISORDER " : " ADHD " , " FACEBOOK ADDICTION DISORDER " : " FAD " , " INTERNET GAMING ADDICTION " : " IGA " , " EMOTIONAL INTELLIGENCE " : " EI " , " INTERNET ADDICTION DISORDER " : " IAD " , " INTERNET ADDICTION TEST " : " IAT " , " PAX GOOD BEHAVIOR GAME " : " PAX GBG " , " INTERNET-PORNOGRAPHY-VIEWING DISORDER " : " IPD " , " RANDOMIZED CONTROLLED TRIAL " : " RCT " , " SEEKING SAFETY " : " SS " , " MALE-TRAUMA RECOVERY EMPOWERMENT MODEL " : " M-TREM " , " SUBSTANCE USE DISORDER " : " SUD " , " FETAL ALCOHOL SPECTRUM DISORDERS " : " FASD " , " FUNCTIONAL MAGNETIC RESONANCE IMAGING " : " FMRI " , " OCCASIONAL GAMBLERS " : " OG  " , " TIME PERSPECTIVE " : " TP " , " MAGNETIC RESONANCE IMAGING " : " MRI " , " HEALTHY VOLUNTEERS " : " HV " , " MOBILE PHONE USE " : " MPU " ,  " ONLINE HEALTH INFORMATION " : " OHI " , " SUITABILITY ASSESSMENT OF MATERIALS " : " SAM " , " INFORMATION AND COMMUNICATION TECHNOLOGIES " : " ICTS " , " COMPULSIVE INTERNET USE " : " CIU " , " LIVING FAMILY TREE " : " LFT " , " HYPOTHALAMIC-PITUITARY ADRENERGIC " : " HPA " , " NON-MEDICAL PRESCRIPTION OPIOID USE " : " NMPOU " , " ONLINE SEXUAL ACTIVITIES " : " OSAS " , " FAMILY NURSE PRACTITIONER " : " FNP " , " REVERSAL THEORY " : " RT " , " GRAY MATTER VOLUME " : " GMV " , " EXCESS SOCIAL MEDIA USE " : " ESMU " , " WIRELESS MOBILE DEVICES " : " WMDS " , " STOCK MARKET INVESTMENT " : " SMI " , " STOP-SIGNAL REACTION TIME " : " SSRT " , " ENERGY DRINKS " : " EDS " , " ENERGY DRINK " : " ED " , " GAMBLING IMPACT AND BEHAVIOR STUDY " : " GIBS " , " SOCIAL NETWORKING SITES " : " SNS " , " PATHOLOGIC GAMBLERS " : " PGS " , " NON-PATHOLOGIC GAMBLERS " : " NON-PGS " , " OAK GAMBLING SCREEN " : " SOGS " , " GAMMA-HYDROXYBUTYRATE " : " GHB " , " PROBLEM INTERNET USAGE " : " PIU " , " BINGE DRINKING " : " BD " , " HUMAN CONNECTOME PROJECT " : " HCP " , " MENTAL HEALTH " : " MH " , " EVENT-RELATED POTENTIALS " : " ERPS " , " VIRTUAL MORRIS WATER TASK " : " VMWT " , " INJECTION DRUG USERS " : " IDUS " , " THALAMIC NEURON THEORY " : " TNT " , " CENTRAL NERVOUS SYSTEM " : " CNS " , " CLIENT SATISFACTION QUESTIONNAIRE " : " CSQ-18 " , " POST-TRAUMATIC STRESS DISORDER " : " PTSD " , " PROBLEMATIC SMARTPHONE USE " : " PSU " , " PROBLEMATIC VIDEO GAME USE " : " PVGU " , " ATTENTION DEFICIT HYPERACTIVITY DISORDER " : " ADHD " , " ATTENTION DEFICIT DISORDER WITH OR WITHOUT HYPERACTIVITY " : " ADHD " , " INFORMATION AND COMMUNICATIONS TECHNOLOGIES " : " ICT " , " GAMBLING MOTIVES QUESTIONNAIRE-REVISED " : " GMQ-R " , " PROBLEMATIC VIDEO GAME PLAY " : " PVGP " , " ADHD SELF-REPORT SCALE " : " ASRS " , " PROBLEM VIDEO GAME PLAYING TEST " : " PVGT " , " SOCIAL MEDIA SITES " : " SMS " , " AMERICAN PSYCHIATRIC ASSOCIATION " : " APA " , " HEALTHY CONTROL " : " HC " , " GAME TRANSFER PHENOMENA " : " GTP " , " PROBLEM VIDEO GAME PLAYING " : " PVP " , " INTERNET PORNOGRAPHY " : " IP " , " MULTIPLAYER ONLINE BATTLE ARENA " : " MOBA " , " GAME ADDICTION SCALE " : " GAS " , " TOBACCO USE DISORDERS " : " TUDS " , " FOOD AND DRUG ADMINISTRATION " : " FDA " , " BEHAVIORAL ADDICTIONS " : " BA " , " TRANSCRANIAL DIRECT CURRENT STIMULATION " : " TDCS " ,  " ALCOHOL DEPENDENCE " : " AD " , " PREFRONTAL CORTEX " : " PFC " , " SERIOUS EDUCATIONAL GAMES " : " SEG " , " ALCOHOL-USE DISORDERS " : " AUDS " , " RESTING-STATE FUNCTIONAL CONNECTIVITY " : " RSFC " , " POSTERIOR CINGULATE CORTEX " : " PCC " , " VIDEO GAME USE " : " VGU " , " NON-VIDEO GAME USE " : " NON-VGU " , " VIDEO GAME DEPENDENCY TEST " : " VDT " , " ORBITOFRONTAL CORTEX " : " OFC " , " SEDENTARY VIDEO GAME " : " SVG " , " EXCESSIVE INTERNET VIDEO GAME PLAY " : " EIGP " , " RELATIVE REINFORCING VALUE " : " RRV " , " DIAGNOSTIC STATISTICAL MANUAL FOR MENTAL DISORDER " : " DSM " , " GAMING DISORDER " : " GD " , " INTERNATIONAL CLASSIFICATION OF DISEASE" : " ICD " , " VIDEO GAME" : " VG " , " PROBLEMATIC VIDEO GAMING" : " PVG " ,  " GAMING ADDICTION " : " GA " , " ONLINE GAMING ADDICTION " : " OGA " , " FUNCTIONAL CONNECTIVITY " : " FC " , " YOUNG'S INTERNET ADDICTION SCALE " : " YIAS" }
# Dictionary of list of abreviation wrotten rightly
update_dic = {"BCI" : "BRAIN-COMPUTER INTERFACE " , " BRAIN-COMPUTER INTERFACES " : "BRAIN-COMPUTER INTERFACE " , " DNNs " : "DEEP NEURAL NETWORK ", "DEEP NEURAL NETWORKS " : "DEEP NEURAL NETWORK " }

abb_dict = dict([(value, key) for key, value in tmp_dic.items()])
abb_dict.update(update_dic)

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

# Swap two dictionaries
data_words = [replace_all(doc, abb_dict) for doc in data]

# Convert sentences to words in the text data
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words2 = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words2, min_count=2, threshold=40) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words2], threshold=40)
fourgram = gensim.models.Phrases(trigram[data_words2], threshold=40)
fivegram = gensim.models.Phrases(fourgram[data_words2], threshold=40)
 

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]
def make_fourgrams(texts):
    return [fourgram[trigram[bigram[doc]]] for doc in texts]
def make_fivegrams(texts):
    return [fivegram[fourgram[trigram[bigram[doc]]]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Stopwords with nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Add some words about research abstract into the list of stopwords
stop_words.extend(["thus","sometimes", "backgrounds", "background", "objectives", "objective", "conclusions", "results", "result", "methods", "method", "design", "methodology", "approach", "purpose", "results", "conclusion", "originality", "value", "findings", "question", "research", "implications", "limitations" 'nevertheless',',',"'",'ha', ',', 'u202f', 'wa', 'le', 'une', 'en', 'et', 'les', 'des', 'la',  'abstract', 'research','may','chapter','also','use', 'result', 'Abstract','Introduction','introduction','background', 'Background','method','Method','conclusion','Conclusion'])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words2)

# Form Bigrams
data_words_bigrams = make_trigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, verb
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(id2word[0])
print(corpus[:1])

# Store the corpus as pickle
import pickle
with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
'''

# Topic Analysis: LDA model
import pandas as pd
import numpy as np
from pprint import pprint

# Gensim
import gensim
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Build LDA model (9topics)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=9, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDAvis_topics.html')

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Find the most discussed topic in each document and tag as 'topic_number' column into each document(paper)
gamma, _ = lda_model.inference(corpus)

df["topic_number"] = ""
df["topic_number"] = [np.argmax(i) for i in gamma]

# The distribution of topics
df["topic_number"].value_counts()

# Write dataframe into excel file for the further analysis
# (Visualize the topic distribution, especially 4 topics higly related 'game' which we define as a structured form of play, usually undertaken for entertainment or fun, and sometimes used as an educational tool)
df_game = df[['date','abstract','topic_number']]
writer = pd.ExcelWriter('paper_game_lda.xlsx')
df_game.to_excel(writer,'Sheet1')
writer.save()
