import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Diferenca mensagem de sistema de usuario
from langchain_core.messages import SystemMessage, HumanMessage

from pydantic import BaseModel, Field
from typing import Literal, List, Dict

load_dotenv()
GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY")) 

TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)


class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER" , "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    api_key=GOOGLE_API_KEY
)

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([

        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()


testes = [
    "Posso reembolsar a internet?",
    "Posso reembolsar, cursos e treinamentos da alura",
    "Quantos graos de areia tem na praia de Salvador?",
    "Quero tirar as minhas ferias, e necessito de uma posicao urgente",
]
    
for msg in testes:
    print(f"Pergunta: {msg}\n-> Resposta: {triagem(msg)}")