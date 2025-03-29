import asyncio
import nest_asyncio
import streamlit as st
import json
import pandas as pd
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

# Permet de faire tourner des boucles asyncio dans Streamlit
nest_asyncio.apply()

# Nettoie la réponse JSON retournée sous forme de chaîne (retire les balises Markdown)
def clean_json_response(response):
    if isinstance(response, str):
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].lstrip()
        if response.endswith("```"):
            response = response[:-len("```")].rstrip()
    return response

# Fonction asynchrone pour interroger l'agent Azure OpenAI
async def run_agent(user_input: str) -> str:
    async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
        # Récupération de la définition de l'agent depuis Azure
        agent_definition = await client.agents.get_agent(agent_id="asst_E5nFroutEcRYyKkXsLkMwPvJ")
        agent = AzureAIAgent(client=client, definition=agent_definition)
        thread: AzureAIAgentThread = None
        try:
            # Envoie le message et récupère la réponse de l'agent
            response = await agent.get_response(messages=user_input, thread=thread)
            return str(response)
        finally:
            # Nettoyage du thread si nécessaire
            if thread:
                await thread.delete()

# Fonction synchrone qui enveloppe l'appel à l'agent, adaptée pour Streamlit
def get_agent_response(user_input: str) -> list:
    raw_response = asyncio.run(run_agent(user_input))
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

# Fenêtre de dialogue Streamlit affichant les codes validés
@st.dialog("PDF viewer", width="large")
def show_validation_dialog():
    validated_rows = st.session_state.get("validated_rows_temp", [])
    st.write("Codes envoyés à l'application **Hopital Management**")
    if validated_rows:
        st.markdown("### Liste récapitulative des codes validés")
        validated_codes = [row["code"] for row in validated_rows]
        df_codes = pd.DataFrame(validated_codes, columns=["Code Validé"])
        st.table(df_codes)
    else:
        st.warning("Aucun code n'a été validé.")

# Initialisation du session_state si non présent
if "agent_response" not in st.session_state:
    st.session_state.agent_response = None

# Barre latérale avec boutons de navigation (fictifs pour l'instant)
with st.sidebar:
    st.button("Logo")
    st.button("Direct")
    st.button("Language")

# Titre principal et champ de saisie pour les notes du médecin
st.title("Harmattan AI Interface")
doctor_notes = st.text_area("Mettez vos notes ici", height=150)

# Bouton d'envoi des notes au modèle IA
if st.button("Envoyer"):
    if doctor_notes.strip():
        with st.spinner("Envoi de la demande à l'agent Harmattan..."):
            st.session_state.agent_response = get_agent_response(doctor_notes)
    else:
        st.warning("Veuillez saisir les notes du médecin.")

# Affichage des résultats si une réponse est disponible
if st.session_state.agent_response:
    st.subheader("Réponse de l'agent")
    df_response = pd.DataFrame(st.session_state.agent_response)

    # Affichage de chaque ligne de résultat dans une grille de colonnes
    for idx, row in df_response.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 1, 2])

        with col1:
            st.markdown(f"**Extrait**<br>{row['extract']}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Catégorie**<br>{row['category']}", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**Code**<br>{row['code']}", unsafe_allow_html=True)
        with col4:
            if row.get("url"):
                st.markdown(f"[Lien]({row['url']})")
        with col5:
            checkbox_key = f"validation_{idx}"
            st.checkbox("Valider", key=checkbox_key)

        st.markdown("---")  # Séparateur entre chaque bloc

    # Bouton de sauvegarde des validations cochées
    if st.button("Sauvegarder les validations"):
        validated_rows = [row.to_dict() for idx, row in df_response.iterrows()
                          if st.session_state.get(f"validation_{idx}", False)]
        st.session_state.validated_rows_temp = validated_rows
        show_validation_dialog()