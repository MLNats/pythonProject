import streamlit as st
import pandas as pd
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import imagehash
import time
import hyperlink

st.title("EdTech Search Engine for Text and Images in Video content.")
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def connect2ES():
    # connect to ES on localhost on port 9200
    es = Elasticsearch([{'host': 'host.docker.internal', 'port': 9200, 'use_ssl': False}])
    if es.ping():
            st.header('Connected to Es!')
    else:
            st.header('Could not connect to Es!')
            exit(0)
    return es

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_encoder():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def keywordSearch(q):
    #Search by Keywords
    b={
            'query':{
                'match':{
                    "transcript":q
                }
            }
        }

    res= es.search(index='transcripts-index',body=b)
    # st.header("Keyword Search:")
    #print("*********************************************************************************");

    return res

# Search by Vec Similarity
def sentenceSimilaritybyNN(embed, es, sent):
    query_vector = tf.make_ndarray(tf.make_tensor_proto(embed([sent]))).tolist()[0]
    #print(query_vector)
    b = {"query" : {
                "script_score" : {
                    "query" : {
                        "match_all": {}
                    },
                    "script" : {
                        "source": "cosineSimilarity(params.query_vector, 'transcript_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
             }
        }


    #print(json.dumps(b,indent=4))
    res= es.search(index='transcripts-index',body=b)

    return res

# Search by Vec Similarity
def imageSimilaritybyNN(es, image_hash):
    b = {"query" : {
                "script_score" : {
                    "query" : {
                        "match_all": {}
                    },
                    "script" : {
                        "source": "1 / (1 + l2norm(params.image_vectors, 'image_vector'))",
                        "params": {"image_vectors": image_hash}
                    }
                }
             }
        }


    #print(json.dumps(b,indent=4))
    res= es.search(index='videoframehash-index',body=b)
    return res

es = connect2ES()

encoder_load_state = st.text('Setting up the search Engine.')
embed = load_encoder()
encoder_load_state.text("Engine Ready! ")

def convert_to_preferred_format(sec):
   #print("GIven Input " + str(sec))
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   #print("seconds value in hours:",hour)
   #print("seconds value in minutes:",min)
   #print(hour,min,sec)
   return "%02d:%02d:%02d" % (hour, min, sec)

def search_data(es, embed, query):
    res = keywordSearch(query)
    required_df = {"Video_Link": [], "From_time_point": [], "Transcript": [], "score": []}
    for hit in res['hits']['hits']:
        video_link = "https://www.youtube.com/watch?v="
        video_link += hit['_source']['video_id'].rstrip("&list.json").lstrip("youtube_")
        required_df["Video_Link"].append(str(hyperlink.parse(video_link)))
        required_df["From_time_point"].append(convert_to_preferred_format(hit['_source']['start']))
        #required_df["To_time_point"].append(hit['_source']['end']/60)
        required_df["Transcript"].append(hit['_source']['transcript'])
        required_df["score"].append(hit['_score'])

    final_data = pd.DataFrame(required_df)

    final_data['normalized_score'] = 0.4 * ((final_data["score"] - final_data['score'].min()) / \
    (final_data["score"].max() - final_data['score'].min()))

    res = sentenceSimilaritybyNN(embed, es, query)

    required_df = {"Video_Link": [], "From_time_point": [], "Transcript": [], "score": []}

    for hit in res['hits']['hits']:
        video_link = "https://www.youtube.com/watch?v="
        video_link += hit['_source']['video_id'].rstrip("&list.json").lstrip("youtube_")
        required_df["Video_Link"].append(str(hyperlink.parse(video_link)))
        required_df["From_time_point"].append(convert_to_preferred_format(hit['_source']['start']))
        #required_df["To_time_point"].append(hit['_source']['end']/60)
        required_df["Transcript"].append(hit['_source']['transcript'])
        required_df["score"].append(hit['_score'])

    final_data_sem = pd.DataFrame(required_df)

    final_data_sem['normalized_score'] = 0.6 * ((final_data_sem["score"] - final_data_sem['score'].min()) / \
                                     (final_data_sem["score"].max() - final_data_sem['score'].min()))
    full_req_data = pd.concat([final_data, final_data_sem])
    full_req_data.sort_values(by='normalized_score', ascending=False, inplace=True)
    return full_req_data


search_type = st.radio(
    "What type of search do you wish to perform?",
    ('Text', 'Image'))
if search_type == 'Text':
    query_text = st.text_input('Enter the query Text to Search', 'How to install Python?')
    if st.checkbox('Generate search Results'):
        st.subheader('Search results')
        data_load_state = st.text('Searching for the best results...')
        data = search_data(es,embed,query_text)
        data_load_state.text("Done! ")
        st.dataframe(data)
else:
    uploaded_file = st.file_uploader("Choose a Image file to upload (png or jpg)")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        hash_value = imagehash.phash(img, 16)
        req_value = list(1.0 * hash_value.hash.flatten())
        if st.checkbox('Generate relevant Results'):
            st.subheader('Search results')
            data_load_state_new = st.text('Searching for the best results...')
            res = imageSimilaritybyNN(es, req_value)
            required_df = {"Video_Link": [], "From_time_point": [], "score": []}
            for hit in res['hits']['hits']:
                required_df['Video_Link'].append(str(hit['_source']['video_link']))
                required_df["From_time_point"].append(str(hit['_source']['start_time']))
                required_df["score"].append(hit['_score'])
            data_load_state_new.text("Done! ")
            st.dataframe(pd.DataFrame(required_df))

