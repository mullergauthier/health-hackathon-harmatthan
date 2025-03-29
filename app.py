import asyncio
import streamlit as st
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

# --- Partie Back-end : fonction asynchrone pour interagir avec l'agent Azure ---
async def run_agent(user_input: str) -> str:
    async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
        # Remplacez "asst_E5nFroutEcRYyKkXsLkMwPvJ" par l'ID réel de votre agent
        agent_definition = await client.agents.get_agent(agent_id="asst_E5nFroutEcRYyKkXsLkMwPvJ")
        agent = AzureAIAgent(client=client, definition=agent_definition)
        thread: AzureAIAgentThread = None
        try:
            # Appel de l'agent avec le texte en entrée
            response = await agent.get_response(messages=user_input, thread=thread)
            # On retourne le texte de la réponse (ou conversion en string selon la structure de response)
            return str(response)
        finally:
            if thread:
                await thread.delete()

def get_agent_response(user_input: str) -> str:
    """
    Fonction synchrone qui exécute la fonction asynchrone run_agent.
    Attention : dans certains environnements Streamlit déjà asynchrones,
    il peut être nécessaire d'adapter la gestion de l'event loop.
    """
    return asyncio.run(run_agent(user_input))

# --- Partie Front-end : Interface Streamlit ---
# Sidebar (boutons placeholders)
with st.sidebar:
    st.button("Logo")
    st.button("Direct")
    st.button("Language")
    st.button("LLM 1")
    st.button("LLM 2")

st.title("Doctor Input Interface")

# Zone de saisie pour les notes du médecin
doctor_notes = st.text_area("Doctor input (notes)", height=150)

# Bouton pour envoyer la demande à l'agent Azure
if st.button("Envoyer"):
    if doctor_notes:
        st.info("Envoi de la demande à l'agent Azure...")
        # Appel de l'agent et affichage de la réponse
        response_text = get_agent_response(doctor_notes)
        st.subheader("Réponse de l'agent")
        st.write(response_text)
    else:
        st.warning("Veuillez saisir les notes du médecin.")



