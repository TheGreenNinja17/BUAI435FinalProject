import os
import importlib
import openai
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Load models and data
w2v_model = Word2Vec.load("w2v_model.bin")
working_data = pd.read_pickle("working_data.pkl")
aggregated_data = pd.read_pickle("aggregated_data.pkl")
product_similarity_sparse = joblib.load("product_similarity_sparse.pkl")
user_product_sparse_csr = joblib.load("user_product_sparse_csr.pkl")

with open('svd_model.pkl', 'rb') as file:
    svd = pickle.load(file)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model):
    tokens = text.split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def compute_cosine_similarity(product_embedding, user_embedding): 
    if np.linalg.norm(product_embedding) > 0 and np.linalg.norm(user_embedding) > 0:
        return cosine_similarity([product_embedding], [user_embedding])[0][0] 
    else:
        return 0

def recommend_similar_products(product_id, product_similarity_sparse, product_id_mapping, top_n=5):
    reverse_mapping = {v: k for k, v in product_id_mapping.items()}
    product_index = reverse_mapping.get(product_id)
    if product_index is None:
        return []
    product_similarities = product_similarity_sparse[product_index].toarray().flatten()
    similar_indices = np.argsort(product_similarities)[::-1][1:top_n+1]
    similar_products = [product_id_mapping[idx] for idx in similar_indices]
    return similar_products

def recommend_for_user(customer_id, user_product_sparse, product_similarity_sparse, customer_id_mapping, product_id_mapping, top_n=5):
    reverse_customer_mapping = {v: k for k, v in customer_id_mapping.items()}
    user_index = reverse_customer_mapping.get(customer_id)
    if user_index is None:
        return []
    user_interactions = user_product_sparse[user_index].toarray().flatten()
    interacted_products = np.where(user_interactions > 0)[0]
    scores = np.zeros(product_similarity_sparse.shape[0])
    for product_idx in interacted_products:
        scores += product_similarity_sparse[product_idx].toarray().flatten()

    scores[interacted_products] = 0
    recommended_indices = np.argsort(scores)[::-1][:top_n]
    recommended_products = [product_id_mapping[idx] for idx in recommended_indices]
    return recommended_products

product_id_mapping = dict(enumerate(working_data['product_id'].astype('category').cat.categories))
customer_id_mapping = dict(enumerate(working_data['customer_id'].astype('category').cat.categories))

def get_top_5_products_for_user(user_id):
    unique_products = working_data['product_id'].unique()
    ratings = []
    for product in unique_products:
        pred = svd.predict(user_id, product)
        ratings.append([product, pred.est])
    ratings_df = pd.DataFrame(ratings, columns=['product_id', 'predicted_rating'])
    top_ratings = ratings_df.sort_values(by='predicted_rating', ascending=False).head(5)
    top_products = top_ratings.merge(aggregated_data[['product_id', 'product_title']], on='product_id', how='left')
    return top_products

def get_global_top_5():
    if 'star_rating' in working_data.columns:
        top_global = working_data.groupby('product_id')['star_rating'].mean().sort_values(ascending=False).head(5)
        top_prods = top_global.index.tolist()
        df_top = aggregated_data[aggregated_data['product_id'].isin(top_prods)][['product_id','product_title']]
        return df_top
    else:
        return get_top_5_products_for_user(user_id=working_data['customer_id'].iloc[0])

SYSTEM_PROMPT = (
    "You are a helpful, fun, and friendly assistant. You are always to-the-point, "
    "but you present recommendations with a warm and cheerful tone."
)

def chatgpt_respond(message, role="user"):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":role,"content":message}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

st.title("Fun & Friendly Recommendation Chatbot")

if 'state' not in st.session_state:
    st.session_state.state = 'initial'

user_input = st.text_input("You: ", value="", key="user_input")
if st.button("Send"):
    if st.session_state.state == 'initial':
        prompt = ("Hey there! How can I help you today?\n"
                  "Do you want:\n"
                  "1) Products similar to a given product\n"
                  "2) Recommendations based on a user ID\n"
                  "3) The top five products overall?\n"
                  "Just let me know which one!")
        response = chatgpt_respond(prompt)
        st.write("ChatGPT: " + response)
        st.session_state.state = 'await_choice'

    elif st.session_state.state == 'await_choice':
        # Use ChatGPT to classify the user input
        user_query = user_input.strip()
        classification_prompt = (
            "The user said:\n"
            f"\"{user_query}\"\n\n"
            "Classify this request into one of the following categories:\n"
            "1) 'similar' if they want products similar to another product\n"
            "2) 'user' if they want recommendations based on a user_id\n"
            "3) 'top' if they want the top five products overall.\n\n"
            "Respond with ONLY ONE WORD: SIMILAR, USER, or TOP."
        )
        classification_response = chatgpt_respond(classification_prompt)
        classification = classification_response.upper().strip()

        if "SIMILAR" in classification:
            st.session_state.state = 'ask_product_id'
            resp = chatgpt_respond("Awesome! Could you give me the product_id you want to find similar products to?")
            st.write("ChatGPT: " + resp)

        elif "USER" in classification:
            st.session_state.state = 'ask_user_id'
            resp = chatgpt_respond("Great! Please provide your user_id so I can personalize some recommendations.")
            st.write("ChatGPT: " + resp)

        elif "TOP" in classification:
            st.session_state.state = 'show_top_5'
            top5 = get_global_top_5()
            product_list = "\n".join([f"- {row['product_title']} (ID: {row['product_id']})" for i, row in top5.iterrows()])
            message = f"Here are the top 5 products overall:\n{product_list}\n\nWould you like another recommendation task?"
            resp = chatgpt_respond(message)
            st.write("ChatGPT: " + resp)
            st.session_state.state = 'another_task'
        else:
            resp = chatgpt_respond("I'm sorry, I couldn't determine what you wanted. Could you please rephrase?")
            st.write("ChatGPT: " + resp)

    elif st.session_state.state == 'ask_product_id':
        product_id = user_input.strip()
        recs = recommend_similar_products(product_id, product_similarity_sparse, product_id_mapping)
        if recs:
            recommendations_str = ""
            for r in recs:
                title = aggregated_data.loc[aggregated_data['product_id'] == r, 'product_title']
                if not title.empty:
                    recommendations_str += f"- {title.values[0]} (ID: {r})\n"
                else:
                    recommendations_str += f"- Product ID: {r}\n"
            prompt = f"Here are products similar to {product_id}:\n{recommendations_str}\n\nWould you like another recommendation task?"
            resp = chatgpt_respond(prompt)
            st.write("ChatGPT: " + resp)
            st.session_state.state = 'another_task'
        else:
            resp = chatgpt_respond("Hmm, I couldn't find that product. Can you try a different product_id?")
            st.write("ChatGPT: " + resp)

    elif st.session_state.state == 'ask_user_id':
        user_text = user_input.strip()
        try:
            user_id = int(user_text)
            recs = recommend_for_user(user_id, user_product_sparse_csr, product_similarity_sparse, customer_id_mapping, product_id_mapping)
            if recs:
                recommendations_str = ""
                for r in recs:
                    title = aggregated_data.loc[aggregated_data['product_id'] == r, 'product_title']
                    if not title.empty:
                        recommendations_str += f"- {title.values[0]} (ID: {r})\n"
                    else:
                        recommendations_str += f"- Product ID: {r}\n"
                prompt = f"Here are some recommendations for user {user_id}:\n{recommendations_str}\n\nWould you like another recommendation task?"
                resp = chatgpt_respond(prompt)
                st.write("ChatGPT: " + resp)
            else:
                resp = chatgpt_respond("I didn't find any recommendations for that user. Maybe they haven't rated enough products?\nWould you like another task?")
                st.write("ChatGPT: " + resp)
            st.session_state.state = 'another_task'
        except ValueError:
            resp = chatgpt_respond("That doesn't look like a valid user_id. Please enter a number.")
            st.write("ChatGPT: " + resp)

    elif st.session_state.state == 'another_task':
        answer = user_input.lower().strip()
        if 'yes' in answer or 'another' in answer:
            resp = chatgpt_respond("Awesome! Do you want similar products, user-based recommendations, or the top five products?")
            st.write("ChatGPT: " + resp)
            st.session_state.state = 'await_choice'
        else:
            resp = chatgpt_respond("Alright, thanks for stopping by! Have a fantastic day!")
            st.write("ChatGPT: " + resp)
            st.session_state.state = 'done'
