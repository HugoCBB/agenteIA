import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, List
from Aula2.main import politicas_rag
from Aula1.main import triagem

from IPython.display import display, Image

from langgraph.graph import StateGraph, START, END


load_dotenv()

GOOGLE_API_KEY = str(os.getenv("GOOGLE_API_KEY"))
KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]


class AgentState(TypedDict, total = False):
    mensagem: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str



def node_triagem(state: AgentState) -> AgentState:
    print("Executando no triagem")
    return {"triagem": triagem(state["mensagem"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando no de auto resolver")
    resposta_rag = politicas_rag(state["mensagem"])

    update: AgentState = {
        "resposta": resposta_rag.get("answer"),
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag.get("contexto_encontrado", False),
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando o no de pedir info")
    faultantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ",".join(faultantes) if faultantes else "Tema e contexto especifico"

    return {"resposta": f"Para avancar, preciso que detaalhe: {detalhe}",
            "citacoes": [],
            "acao_final": "PEDIR_INFO"
            }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando no de abrir chamado")
    triagem = state["triagem"]

    return {
        "resposta": f"Abrindo chamado com urgencia {triagem['urgencia']}. Descricao: {state['mensagem'][:140]}",
        "citacoes":[],
        "acao_final":"ABRIR_CHAMADO"
    }


def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto_resolver...")

    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"

    state_da_pergunta = (state["mensagem"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"

    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"



workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()

# graph_bytes = grafo.get_graph().draw_mermaid_png()
# display(Image(graph_bytes))


testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar cursos ou treinamentos da Alura?",
          "É possível reembolsar certificações do Google Cloud?",
          "Posso obter o Google Gemini de graça?",
          "Qual é a palavra-chave da aula de hoje?",
          "Quantas capivaras tem no Rio Pinheiros?"]

for msg_test in testes:
    resposta_final = grafo.invoke({"mensagem": msg_test})

    triag = resposta_final.get("triagem", {})
    print(f"PERGUNTA: {msg_test}")
    print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"RESPOSTA: {resposta_final.get('resposta')}")
    if resposta_final.get("citacoes"):
        print("CITAÇÕES:")
        for citacao in resposta_final.get("citacoes"):
            print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
            print(f"   Trecho: {citacao['trecho']}")

    print("------------------------------------")