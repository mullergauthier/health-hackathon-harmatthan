import asyncio
import nest_asyncio
import streamlit as st
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

nest_asyncio.apply()

# --- Partie Back-end : fonction asynchrone pour interagir avec l'agent Azure ---
async def run_agent(user_input: str) -> str:
    async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
        # Remplacer par votre véritable agent ID Azure
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
def get_agent_response(user_input: str) -> str:
    return asyncio.run(run_agent(user_input))

# --- Interface Streamlit ---
with st.sidebar:
    st.button("Logo")
    st.button("Direct")
    st.button("Language")
    st.button("LLM 1")
    st.button("LLM 2")

st.title("Doctor Input Interface")

doctor_notes = st.text_area("Doctor input (notes)", height=150)

if st.button("Envoyer"):
    if doctor_notes.strip():
        with st.spinner("Envoi de la demande à l'agent Azure..."):
            response_text = get_agent_response(doctor_notes)
        st.subheader("Réponse de l'agent")
        st.write(response_text)
    else:
        st.warning("Veuillez saisir les notes du médecin.")
