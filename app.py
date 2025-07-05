# app.py
import streamlit as st
from utils import PERSONA_NAMES
from rag_components import load_and_preprocess_data, get_retriever, create_agentic_rag_app, generate_suggested_questions

st.set_page_config(page_title="Data Persona Interativa - Tera", page_icon="🎓", layout="wide")

if 'screen' not in st.session_state: st.session_state.screen = 'home'
if 'messages' not in st.session_state: st.session_state.messages = []
if 'agentic_app' not in st.session_state: st.session_state.agentic_app = None

def render_footer():
    st.markdown("---")
    st.markdown("Desenvolvido por [Pedro Costa](https://www.linkedin.com/in/pedrocsta/) | Product Marketing & Martech Specialist")

def render_home_screen():
    st.title("Data Persona Interativa 🎓")
    
    # NOVO TEXTO PARA A TERA
    st.markdown("""
    Esta aplicação cria uma persona interativa e 100% data-driven, representando os alunos e profissionais que compõem o ecossistema da **Tera**. 
    Utilizando a arquitetura **RAG com Agentes de IA**, esta persona responde exclusivamente com base em uma base de conhecimento real (depoimentos, pesquisas, etc.), garantindo insights autênticos.

    Seu verdadeiro poder é a **autonomia** para os times da Tera. Em vez de iniciar um novo ciclo de pesquisa para cada dúvida, a equipe pode conversar diretamente com uma representação fiel de seus alunos para validar hipóteses sobre cursos, testar narrativas de marketing e aprofundar a empatia de forma ágil.
    """)
    with st.expander("⚙️ Conheça o maquinário por trás da mágica"):
        st.markdown("""
        - **Modelo de Linguagem (LLM):** `Google Gemini 1.5 Pro & Flash`
        - **Arquitetura:** `RAG com Agentes de IA (LangGraph)`
        - **Orquestração:** `LangChain`
        - **Interface e Aplicação:** `Python + Streamlit`
        - **Base de Dados Vetorial:** `ChromaDB (in-memory)`
        """)
    st.divider()

    # NOVOS SELETORES
    st.selectbox('Selecione a Escola:', ('Tera',), help="Esta versão é focada na Tera.")
    selected_product = st.selectbox(
        'Selecione a Área de Estudo para a Persona:',
        ('Product Management', 'UX Design', 'Data Analytics')
    )

    if st.button("Iniciar Entrevista", type="primary"):
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Chave da API não configurada.")
            st.stop()
        api_key = st.secrets["GEMINI_API_KEY"]
        with st.spinner("Preparando a persona e seus agentes..."):
            full_data = load_and_preprocess_data("data")
            if full_data.empty: st.error("Nenhum dado válido na pasta 'data'."); st.stop()
            retriever = get_retriever(full_data, selected_product, api_key)
            if retriever is None: st.error(f"Não há dados para a área de '{selected_product}'."); st.stop()
            st.session_state.agentic_app = create_agentic_rag_app(retriever, api_key)
            st.session_state.persona_name = PERSONA_NAMES[selected_product]
            st.session_state.product_name = selected_product
            st.session_state.suggested_questions = generate_suggested_questions(api_key, st.session_state.persona_name, selected_product)
            st.session_state.screen = 'chat'
            st.session_state.messages = []
            st.rerun()
    render_footer()

def handle_new_message(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    payload = {"question": prompt, "chat_history": st.session_state.messages[:-1], "product_name": st.session_state.product_name, "persona_name": st.session_state.persona_name}
    with st.chat_message("assistant"):
        with st.spinner("A equipe de agentes está pensando..."):
            final_state = st.session_state.agentic_app.invoke(payload)
            response_content = final_state.get('final_answer', "Desculpe, não consegui processar uma resposta.")
            source_documents = final_state.get('documents', [])
            st.markdown(response_content)
            if source_documents:
                with st.expander("Ver fontes utilizadas"):
                    for doc in source_documents: st.info(doc.page_content)
    st.session_state.messages.append({"role": "assistant", "content": response_content, "sources": source_documents})

def render_chat_screen():
    st.title(f"Entrevistando: {st.session_state.persona_name}")
    st.markdown("Esta é uma demonstração. Converse com a persona para extrair insights.")
    st.divider()

    # Exibe o histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Ver fontes utilizadas"):
                    for doc in message["sources"]: st.info(doc.page_content)

    # Input do chat
    if prompt := st.chat_input("Digite para conversar!"):
        handle_new_message(prompt)
        st.rerun()

    st.divider()
    if st.button("⬅️ Iniciar Nova Entrevista"):
        keys_to_clear = ['messages', 'agentic_app', 'persona_name', 'product_name', 'suggested_questions']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.session_state.screen = 'home'
        st.rerun()
    render_footer()

# --- Lógica Principal ---
if st.session_state.screen == 'home':
    render_home_screen()
elif st.session_state.screen == 'chat':
    render_chat_screen()
