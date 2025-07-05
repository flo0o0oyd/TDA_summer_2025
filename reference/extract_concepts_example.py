# -*- coding: utf-8 -*-

"""compute_field_social_science_articles_abstracts_concept_nodes.py: Pull nodes 
of concepts from WoS abstracts on social science."""					  

__author__ = "Russell J. Funk"
__date__ = "February 8, 2021"

# built in modules
import collections
import time

# third party modules
import pandas as pd
import swifter
import spacy
import html2text
import pymysql.cursors
import sqlalchemy

# custom modules

# initialize modules
nlp = spacy.load("en_core_web_lg", disable=["ner"]) 

# configuration
MIN_CONCEPT_LEN = 3
MAX_CONCEPT_LEN = 100
SUBJECT_EXTENDED_MIN_NRECORD_ID_FREQ = 2
DATABASE_TABLE_NAME = "field_social_science_articles_abstracts_concept_nodes"
CONNECTION_STRING = "mysql+pymysql://root:russell0@localhost/wos_2017"

# tokens to drop
DROP_TOKENS = {"a", "an", "and", "another", "any", "both", "enough", "her", "his", "its",
              "many", "most", "much", "my", "or", "other", "our", "some", "that", "the",
              "their", "these", "this", "those", "what", "whatever", "which", "whichever",
              "whose", "your", "-PRON-"} # note, -PRON- is how spacy lemmatizes pronouns
DROP_TOKENS = {token.lower() for token in DROP_TOKENS} # need to be lower case for comparisons

# punctuation to drop
DROP_PUNCTUATION = '}\'!([$):%];,&`?.{~-@"#_'
DROP_PUNCTUATION_TRANSLATION = str.maketrans(DROP_PUNCTUATION, " "*len(DROP_PUNCTUATION))

# special concepts to drop (data source specific things to drop)
# DROP_SPECIAL_CONCEPTS = {"American Physical Society",}
# DROP_SPECIAL_CONCEPTS = {concept.lower() for concept in DROP_SPECIAL_CONCEPTS} # need to be lower case for comparisons

def open_connection():
  """connect to MySQL server"""
  conn_ = pymysql.connect(host = "localhost",
                          user = "root",
                          password = "russell0",
                          charset = "utf8mb4",
                          use_unicode = True,
                          cursorclass = pymysql.cursors.SSCursor)
  return conn_

def main():


  print("getting subjects_extended...")
  conn = open_connection()
  subjects_extended = pd.read_sql(""" select distinct subject_extended
                                         from wos_2017.field_social_science_articles
                                         where abstract is not null;""", con=conn) 
  conn.close()

  # loop over subjects_extended
  for subject_extended in subjects_extended.subject_extended:

    print("working on %s..." % (subject_extended))
  
    print("pulling the abstracts from MySQL...")
    conn = open_connection()
    df = pd.read_sql(""" select record_id,
                                subject_extended,
                                abstract
                         from wos_2017.field_social_science_articles
                         where abstract is not null
                         and subject_extended = %s;""", params=(subject_extended,), 
                                         con=conn) 
    conn.close()























  
    print("subsetting to remove abstracts with no content...")
    df = df.dropna(subset=["abstract"])
    df = df[df.abstract != "<p></p>"]
    df = df[df.abstract != "<p> </p>"]
  
    print("converting abstracts to string...")
    df.abstract = df.abstract.astype(str)
  
    print("removing html...")
    df.abstract = df.abstract.swifter.progress_bar(enable=True).apply(html2text.html2text)
  
    print("apply spacy nlp...")
    df.abstract = df.abstract.swifter.progress_bar(enable=True).apply(nlp)
    #start = time.process_time()
    #df.abstract = [doc for doc in nlp.pipe(df.abstract, batch_size=100)]
    #print(time.process_time() - start)
  
    print("pulling raw noun chunks...")
    df = df.assign(abstract = df.abstract.swifter.progress_bar(enable=True).apply(lambda doc: [chunk for chunk in doc.noun_chunks]))  
    print("creating raw 2 mode edge list...")
    df = df.abstract.swifter.progress_bar(enable=True).apply(pd.Series).merge(df, 
                                                                              left_index = True, 
                                                                              right_index = True).drop(["abstract"], axis = 1)  
    
    print("melting the data frame...")
    df = (df.melt(id_vars = ["record_id", "subject_extended"])
            .drop(["variable"], axis = 1)
            .rename(columns={"value": "concept"})
            .dropna())
    
    print("pulling lemmas...")
    df.concept = df.concept.swifter.progress_bar(enable=True).apply(lambda concept: [token.lemma_ for token in concept])
      
    print("joining list to string...")
    df.concept = df.concept.str.join(" ") 
    
    print("converting to lower case...")
    df.concept = df.concept.str.lower() 
  
    print("replacing tokens from DROP_TOKENS...") # should come before dropping of punctuation
    df.concept = df.concept.swifter.progress_bar(enable=True).apply(lambda concept: " ".join([token for token in concept.split() if token.lower() not in DROP_TOKENS])) 
      
    print("replacing punctuation from DROP_PUNCTUATION...")
    df.concept = df.concept.str.translate(DROP_PUNCTUATION_TRANSLATION) 
    
    print("dropping concepts that do not contain letters...")
    df = df[df.concept.str.contains("[a-zA-Z]", regex=True)]
  
    print("final normalization of white space...")
    #df.concept = df.concept.swifter.progress_bar(enable=True).apply(lambda concept: " ".join(concept.split())) 
    df.concept = df.concept.apply(lambda concept: " ".join(concept.split())) 
  
    print("running final trim of leading and trailing white space...")
    df.concept = df.concept.str.strip()
  
    print("dropping concepts that are shorter than MIN_CONCEPT_LEN and longer than MAX_CONCEPT_LEN...")
    df = df[(df.concept.str.len() >= MIN_CONCEPT_LEN) & (df.concept.str.len() <= MAX_CONCEPT_LEN)]
  
    # print("dropping special data source specific concepts from DROP_SPECIAL_CONCEPTS...")
    # df = df[~df.concept.isin(DROP_SPECIAL_CONCEPTS)]
    
    print("dropping missing values...")
    df = df.dropna()
    
    print("adding concept_no...")
    df = df.assign(concept_no = df.groupby(["record_id"]).cumcount())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("add count of occurrences for concept across record_ids for subject_extended...")
    df = df.assign(concept_subject_extended_nrecord_ids=df.groupby("concept")["record_id"].transform("count"))
    
    print("dropping concepts that appear in fewer than SUBJECT_EXTENDED_MIN_NRECORD_ID_FREQ records across the database...")
    df = df[df.concept_subject_extended_nrecord_ids >= SUBJECT_EXTENDED_MIN_NRECORD_ID_FREQ]
    
    print("adding to the database...")
    sql_engine = sqlalchemy.create_engine(CONNECTION_STRING)
    df.to_sql(name=DATABASE_TABLE_NAME, 
              con=sql_engine, 
              schema="wos_2017",
              if_exists="append", 
              index=False)  
    sql_engine.dispose()
  
if __name__ == "__main__":
 main()