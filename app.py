import asyncio
import nest_asyncio
import streamlit as st
import json
import pandas as pd
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

nest_asyncio.apply()

def clean_json_response(response):
    # Si l'entrée est une chaîne, retirez les délimiteurs markdown
    if isinstance(response, str):
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].lstrip()
        if response.endswith("```"):
            response = response[:-len("```")].rstrip()
    return response

# Fonction asynchrone pour récupérer la réponse de l'agent
async def run_agent(user_input: str) -> str:
    async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
        agent_definition = await client.agents.get_agent(agent_id="asst_E5nFroutEcRYyKkXsLkMwPvJ")
        agent = AzureAIAgent(client=client, definition=agent_definition)
        thread: AzureAIAgentThread = None
        try:
            response = await agent.get_response(messages=user_input, thread=thread)
            return str(response)
        finally:
            if thread:
                await thread.delete()

# Wrapper synchrone compatible Streamlit
def get_agent_response(user_input: str) -> list:
    raw_response = asyncio.run(run_agent(user_input))
    # Nettoyage de la réponse brute pour retirer les délimiteurs markdown s'il y en a
    raw_response = clean_json_response(raw_response)
    if not raw_response:
        st.error("La réponse de l'agent est vide.")
        return []
    try:
        json_response = json.loads(raw_response)
        if isinstance(json_response, list):
            return json_response
        else:
            st.error("La réponse n'est pas une liste JSON valide.")
            return []
    except json.JSONDecodeError as e:
        st.error(f"Erreur lors du parsing JSON : {e}")
        st.text_area("Réponse brute de l'agent :", raw_response, height=200)
        return []

# Interface Streamlit
with st.sidebar:
    st.button("Logo")
    st.button("Direct")
    st.button("Language")

st.title("Harmattan AI Interface")

doctor_notes = st.text_area("Put your notes here", height=150)

if st.button("Envoyer"):
    if doctor_notes.strip():
        with st.spinner("Envoi de la demande à l'agent Harmattan...",show_time=True):
            response_json_list = get_agent_response(doctor_notes)
        if response_json_list:
            st.subheader("Réponse de l'agent (Tableau)")
            # Conversion de la réponse JSON en DataFrame
            df_response = pd.DataFrame(response_json_list)
            # Si la colonne 'url' existe, créer une nouvelle colonne 'Lien' avec un icône cliquable
            if "url" in df_response.columns:
                df_response['Lien'] = df_response['url'].apply(
                    lambda x: f'<a href="{x}" target="_blank" style="text-decoration:none;">🔗</a>'
                )
                # Optionnel : supprimer la colonne 'url'
                df_response = df_response.drop(columns=["url"])
            # Affichage du DataFrame sous forme de tableau HTML pour permettre le rendu du HTML
            st.markdown(df_response.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.warning("La réponse de l'agent est vide ou invalide.")
    else:
        st.warning("Veuillez saisir les notes du médecin.")
