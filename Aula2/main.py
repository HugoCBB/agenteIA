import os
from typing import Dict, List

import re, pathlib

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

from pathlib import Path

load_dotenv()
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    api_key=GOOGLE_API_KEY,
)

docs  = []

def extract_document():
    for document in Path("../data/pdf/").glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(document))
            docs.extend(loader.load())
            print(f"Documento {document.name} carregado com sucesso")
        except Exception as e:
            print(f"Erro ao carregar arquivo {document.name}")
            
            

    print(f"Total de documentos carregados {len(docs)}")


extract_document()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

chuncks = splitter.split_documents(docs)

for chunck in chuncks:
    print(chunck, "\n")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key= GOOGLE_API_KEY,
)


vectorestore = FAISS.from_documents(chuncks, embeddings)
retriever = vectorestore.as_retriever(search_type="similarity_score_threshold",
                                     search_kwargs={"score_threshold":0.3, "k": 4})

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se Nao houver base suficiente, responda apenas 'Nao sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm, prompt_rag)



def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]




def politicas_rag(pergunta: str) -> Dict:
    docs_relacionados =retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Nao sei.",
                "citacoes": [],
                "contexto_encontrado": False}
    
    answer = document_chain.invoke({"input": pergunta,"context": docs_relacionados})

    txt = (answer or "").strip()

    if txt.strip(".!?") == "Nao sei":
        return {"answer": "Nao sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar cursos ou treinamentos da Alura?",
          "Quantas capivaras tem no Rio Pinheiros?"]

for msg_teste in testes:
    resposta = politicas_rag(msg_teste)
    print(f"PERGUNTA: {msg_teste}")
    print(f"RESPOSTA: {resposta['answer']}")
    if resposta['contexto_encontrado']:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")