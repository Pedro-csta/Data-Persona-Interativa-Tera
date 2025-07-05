# rag_components.py

import os 
import pandas as pd
from typing import TypedDict, List
from streamlit import cache_data, cache_resource

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# FIX for ChromaDB/SQLite on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class AgentState(TypedDict):
    question: str
    chat_history: list
    product_name: str
    persona_name: str
    documents: List[Document]
    search_queries: List[str]
    final_answer: str

class DecomposedQuery(BaseModel):
    search_queries: List[str] = Field(description="Uma lista de 2 a 3 strings de busca otimizadas.")

@cache_data
def load_and_preprocess_data(folder_path):
    all_dataframes = []
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError: return pd.DataFrame()

    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                # CORREÇÃO CRÍTICA AQUI: Adicionado sep=';' para ler o arquivo corretamente
                df = pd.read_csv(file_path, sep=';')
                
                if "text" in df.columns and "product" in df.columns:
                    if filename == 'info_oficial.csv':
                        df['text'] = '[FONTE OFICIAL]: ' + df['text'].astype(str)
                    else:
                        df['text'] = '[OPINIÃO DE USUÁRIO]: ' + df['text'].astype(str)
                    all_dataframes.append(df)
                else:
                    print(f"-> AVISO: Arquivo '{filename}' ignorado. Colunas 'text' e 'product' não encontradas (verifique se o separador é ';').")
            except Exception as e: print(f"Erro ao ler '{filename}': {e}")
            
    if all_dataframes: return pd.concat(all_dataframes, ignore_index=True)
    return pd.DataFrame()

@cache_resource(show_spinner=False)
def get_retriever(_dataframe, product_name, api_key):
    product_df = _dataframe[_dataframe['product'].str.lower() == product_name.lower()].copy()
    if product_df.empty: return None
    documents = [Document(page_content=row['text']) for index, row in product_df.iterrows()]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 25})

def query_analyzer_node(state: AgentState, llm):
    prompt = f"""Sua tarefa é atuar como um especialista em buscas. Analise a pergunta do usuário e o histórico da conversa para gerar de 2 a 3 variações de busca otimizadas.\nHistórico: {state['chat_history']}\nPergunta do Usuário: {state['question']}"""
    structured_llm = llm.with_structured_output(DecomposedQuery)
    response = structured_llm.invoke(prompt)
    return {"documents": [], "search_queries": response.search_queries}

def retrieval_node(state: AgentState, retriever):
    all_docs = [doc for query in state["search_queries"] for doc in retriever.invoke(query)]
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return {"documents": list(unique_docs)}

def synthesis_node(state: AgentState, llm):
    prompt_template = f"""Sua única tarefa é atuar como {state['persona_name']}, um(a) profissional buscando se desenvolver na área de '{state['product_name']}'.
    Você deve responder à "PERGUNTA ATUAL" usando as informações do "CONTEXTO" e do "HISTÓRICO DA CONVERSA".
    Seu tom deve ser o de uma pessoa real: em primeira pessoa, coloquial, equilibrado e construtivo. Varie o início das suas respostas.
    REGRA CRÍTICA - HIERARQUIA: Para fatos sobre o produto (cursos, etc), priorize a `[FONTE OFICIAL]`. Para experiências, use `[OPINIÃO DE USUÁRIO]`. Se houver conflito, comente sobre isso.
    Se a informação não estiver disponível, admita que não sabe.
    HISTÓRICO DA CONVERSA: {state['chat_history']}
    CONTEXTO: {state['documents']}
    PERGUNTA ATUAL: {state['question']}
    Sua Resposta Natural (como {state['persona_name']}):"""
    response = llm.invoke(prompt_template)
    return {"final_answer": response.content}

def create_agentic_rag_app(retriever, api_key):
    if retriever is None: return None
    llm_analyzer = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    llm_synthesizer = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.5)
    workflow = StateGraph(AgentState)
    workflow.add_node("query_analyzer", lambda state: query_analyzer_node(state, llm_analyzer))
    workflow.add_node("retriever", lambda state: retrieval_node(state, retriever))
    workflow.add_node("synthesizer", lambda state: synthesis_node(state, llm_synthesizer))
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()

@cache_data(show_spinner=False)
def generate_suggested_questions(api_key, persona_name, product_name):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
    prompt = f"""Atue como um Pesquisador de UX. Crie 10 perguntas abertas para um(a) aluno(a) interessado(a) em '{product_name}' para descobrir insights sobre perfil, dores e motivações de carreira. Retorne como uma lista Python."""
    try:
        response = llm.invoke(prompt)
        return eval(response.content)
    except Exception as e: print(f"⚠️ Aviso: Falha ao gerar perguntas. Usando fallback. Erro: {e}")
    fallback_questions = {
        "Product Management": ["O que te motivou a buscar uma carreira em Gestão de Produto?", "Qual seu maior receio ou desafio ao pensar em fazer uma transição de carreira para Produto?", "O que você mais valoriza em um curso: professores renomados, networking ou conteúdo prático?", "Descreva uma situação profissional que te fez pensar 'eu preciso aprender mais sobre produto'.", "Como você se imagina aplicando os conhecimentos de produto no seu trabalho atual ou futuro?", "Qual a sua maior dificuldade para montar um portfólio de produto sem ter experiência prévia?", "Que tipo de empresa você sonha em trabalhar como PM?", "Além do conhecimento técnico, que habilidade comportamental você mais quer desenvolver?", "Como você se mantém atualizado sobre as tendências do mercado de produto?", "Se você pudesse perguntar algo para uma Gerente de Produto Sênior, o que seria?"],
        "UX Design": ["O que te atraiu na área de UX Design? Foi a parte visual, a pesquisa com usuários ou a resolução de problemas?", "Qual a sua maior dificuldade hoje para construir um portfólio de UX que chame a atenção?", "Como você lida com a frustração quando um design que você gosta não funciona bem nos testes com usuários?", "Que ferramenta de design (Figma, Sketch, etc.) você mais gosta e por quê?", "Descreva um aplicativo ou site que você considera ter uma experiência de usuário perfeita.", "Qual o seu maior medo ao pensar em apresentar um projeto de design para stakeholders?", "O que você espera que um curso de UX te ensine além de simplesmente mexer nas ferramentas?", "Como você busca empatia com os usuários dos produtos que você desenha?", "Qual a sua maior dúvida sobre o dia a dia de um profissional de UX?", "Se você pudesse redesenhar qualquer produto digital que usa hoje, qual seria e por onde você começaria?"],
        "Data Analytics": ["O que te fez querer trabalhar com dados? Foi a paixão por números, por tecnologia ou por negócios?", "Qual é a sua maior dificuldade ao começar a aprender uma linguagem como Python ou SQL?", "Descreva um momento em que você olhou para um conjunto de dados e se sentiu 'perdido(a)'.", "O que você acha mais fascinante em análise de dados: criar visualizações (gráficos) ou encontrar padrões escondidos?", "Qual a sua maior preocupação sobre o futuro da carreira de análise de dados com a chegada das IAs?", "Como você pretende usar a análise de dados para gerar impacto em uma empresa?", "O que você busca em um curso de dados além de apenas a teoria matemática e estatística?", "Como você lida com uma base de dados 'suja' ou incompleta?", "Qual tipo de problema de negócio você mais gostaria de resolver usando dados?", "Se você pudesse ter um mentor na área de dados, qual seria a primeira pergunta que você faria?"]
    }
    return fallback_questions.get(product_name, fallback_questions["Product Management"])
