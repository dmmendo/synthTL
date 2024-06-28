import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
import utils

class ContextRetriever:
    def __init__(self,passages,mode="RAG",model="gpt-3.5-turbo-0125"):
        assert mode in ["RAG","LLM"]
        self.mode = mode
        self.passages = passages
        if mode == "RAG":
            self.bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            self.bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
            self.corpus_embeddings = self.bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
        elif mode == "LLM":
            self.model = model
            #self.model = "gpt-3.5-turbo-0125"
            #self.model = "gpt-4-0125-preview"
    
    def search(self,query,retrv_top_k=100,rerank_top_k=10):
        if self.mode == "RAG":     
            query = "Query: retrieve the signal descriptions that are used to formalize the phrase \'" + query + "\' into an LTL formula."
            ##### Semantic Search #####
            # Encode the query using the bi-encoder and find potentially relevant passages
            question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
            question_embedding = question_embedding.cuda()
            hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=retrv_top_k)
            hits = hits[0]  # Get the hits for the first query
        
            ##### Re-Ranking #####
            # Now, score all retrieved passages with the cross_encoder
            cross_inp = [[query, self.passages[hit['corpus_id']]] for hit in hits]
            cross_scores = self.cross_encoder.predict(cross_inp)
        
            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]
            
            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            res_passages = [self.passages[hit['corpus_id']] for hit in hits[:rerank_top_k]]
            return res_passages
        elif self.mode == "LLM":
            prompt = "You are being tasked with retrieving phrases within a document that are relevant for a specific phrase. "
            prompt += "The following is the list of phrases which you may use in your response to queries:\n"
            prompt += str(self.passages)
            prompt += "\nResponses must be JSON parsable with the field 'context' which is a JSON parseable list of relevant phrases directly quoted from the above list of passages."
            prompt += "\nNote that you must provide direct quotes in your list from the above list of phrases and the response you provide should contain the top " + str(rerank_top_k) + " most relevant phrases for the query."
            prompt += "Query: retrieve the signal descriptions that are used to formalize the phrase \'" + query + "\' into an LTL formula."
            prompt += "\nAvoid selecting related but irrelevant information as the information provided should be directly used to translating the query phrase to LTL."
            response = utils.get_inference_response(prompt,self.model)
            pred = response["choices"][0]["message"]["content"]
            try:
                res = json.loads(pred)
                res = res["context"]
            except:
                res = []
            return res
            
def get_paragraphs(data_filepath):
    with open(data_filepath,'r') as f:
        raw_data = f.read()
    passages = raw_data.split("\n")
    print("Passages:", len(passages))
    return passages

def get_sentences(data_filepath):
    with open(data_filepath,'r') as f:
        raw_data = f.read()
    passages = raw_data.split(".")
    print("Passages:", len(passages))
    return passages