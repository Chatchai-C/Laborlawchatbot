from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer, util,InputExample
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import numpy as np
import faiss
import requests
import json

#model = SentenceTransformer('bert-base-nli-mean-tokens')
#model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()


cypher_query = '''
MATCH (n) WHERE (n:Greeting OR n:Question) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
greeting_vec = None
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])
    #greeting_corpus = ["สวัสดีครับ","ดีจ้า"]
greeting_corpus = list(set(greeting_corpus))
print(greeting_corpus)  


def compute_similar(corpus, sentence):
    a_vec = model.encode([corpus],convert_to_tensor=True,normalize_embeddings=True)
    b_vec = model.encode([sentence],convert_to_tensor=True,normalize_embeddings=True)
    similarities = util.cos_sim(a_vec, b_vec)
    return similarities


def neo4j_search(neo_query):
    results = run_query(neo_query)
    # Print results
    for record in results:
        response_msg = record['reply']
    return response_msg     


# Function to build FAISS index
def build_faiss_index(corpus):
    # Encode the corpus using the SentenceTransformer model
    corpus_embeddings = model.encode(corpus, convert_to_tensor=False, normalize_embeddings=True)
    # Initialize FAISS index (Flat index for L2 similarity)
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    # Add embeddings to the index
    index.add(corpus_embeddings)
    return index, corpus_embeddings

# Function to compute similar sentences using FAISS
def compute_similar_faiss(corpus_index, sentence, k=1):
    ask_vec = model.encode([sentence], convert_to_tensor=False, normalize_embeddings=True)
    distances, indices = corpus_index.search(ask_vec, k)
    return distances[0], indices[0]

def store_chat_in_neo4j(user_id, sentence, bot_reply):
    # Create a node that contains both the user message and the bot's reply
    cypher_query = '''
    CREATE (c:Chat {userID: $user_id, user_message: $user_message, bot_reply: $bot_reply, timestamp: datetime()})
    '''
    parameters = {"user_id": user_id, "user_message": sentence, "bot_reply": bot_reply}
    run_query(cypher_query, parameters)

# Building FAISS index for the greeting corpus
corpus_index, corpus_embeddings = build_faiss_index(greeting_corpus)

def compute_response(sentence, user_id):

    # Perform similarity search using FAISS
    distances, indices = compute_similar_faiss(corpus_index, sentence)
    
    # If the highest similarity is greater than the threshold, fetch response from Neo4j
    if distances[0] < 0.4:  # FAISS uses L2 distances, so lower is better
        Match_greeting = greeting_corpus[indices[0]]
        My_cypher = f"MATCH (n) WHERE (n:Greeting OR n:Question) AND n.name = '{Match_greeting}' RETURN n.msg_reply AS reply"
        my_msg = neo4j_search(My_cypher)

    else:
     # Check if the sentence is a legal-related question
        legal_keywords = ["กฎหมาย", "แรงงาน", "ศาล", "สัญญา", "สิทธิ", "คำสั่ง", "ระเบียบ", "ข้อบังคับ", "พนักงาน", "นายจ้าง", "ลูกจ้าง", "บริษัท"]
        if all(keyword not in sentence for keyword in legal_keywords):
            # Respond with a message that legal questions cannot be answered
            my_msg = "ขออภัยด้วยครับ แชทบอทตัวนี้สามารถตอบคำถามได้แค่เฉพาะในเชิงกฏหมายเท่านั้น กรุณาถามคำถามที่เฉพาะเจาะจงมากกว่านี้หรือลองเปลี่ยนคำถามดูครับ"
        else:
            OLLAMA_API_URL = "http://localhost:11434/api/generate"
            headers = {
                "Content-Type": "application/json"
            }

            # Prepare the request payload for the TinyLLaMA model
            payload = {
                "model": "supachai/llama-3-typhoon-v1.5",
                "prompt": sentence + "ตอบสั้นๆไม่เกิน 30 คำ",
                "stream": False
            }

            # Send the POST request to the Ollama API
            response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response JSON
                response_data = response.text
                data = json.loads(response_data)
                my_msg = data["response"] + "\nข้อมูลส่วนนี้มาจาก Ollama" # Extract the response from the API
            else:
                # If Ollama API fails, return a default error message
                my_msg = "ขออภัยด้วยครับ แชทบอทไม่สามารถตอบคำถามนี้ได้ กรุณาถามคำถามที่เฉพาะเจาะจงมากกว่านี้หรือลองเปลี่ยนคำถามดูครับ"

    # Store the bot's reply as a single node
    store_chat_in_neo4j(user_id, sentence, my_msg)

    return my_msg

app = Flask(__name__)
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)                   
    try:
        json_data = json.loads(body)                       
        access_token = ''
        secret = ''
        line_bot_api = LineBotApi(access_token)             
        handler = WebhookHandler(secret)                   
        signature = request.headers['X-Line-Signature']     
        handler.handle(body, signature)                     
        msg = json_data['events'][0]['message']['text']     
        tk = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId']  # Get the LINE user ID    
        response_msg = compute_response(msg, user_id)
        line_bot_api.reply_message( tk, TextSendMessage(text=response_msg) )
        print(msg, tk)                                     
    except:
        print(body)                                         
    return 'OK'   
               
if __name__ == '__main__':
    #For Debug
    compute_response("นอนหลับฝันดี","U1234567890abcdef")
    app.run(port=5000)