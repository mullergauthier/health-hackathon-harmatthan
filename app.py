import os
import asyncio
import nest_asyncio
import streamlit as st
import json
import pandas as pd
import logging
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread, AzureAIAgentSettings

# Load secrets from Streamlit secrets.toml

AGENT_ID = st.secrets.azure.AZURE_AI_AGENT_AGENT  # use this in run_agent
os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"] = \
    st.secrets.azure.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = \
    st.secrets.azure.AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME # bridge to use AzureAIAgentSettings.create()

# Configure logger (file + console)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log',
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)
logger = logging.getLogger(__name__)

nest_asyncio.apply()  # allow nested asyncio loops in Streamlit

async def run_agent(user_input: str) -> str:
    # Use DefaultAzureCredential, configured via environment vars from secrets.toml
    async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
        settings = AzureAIAgentSettings.create()
        agent_def = await client.agents.get_agent(agent_id=AGENT_ID)
        agent = AzureAIAgent(client=client, definition=agent_def, settings=settings)
        try:
            return str(await agent.get_response(messages=user_input))
        finally:
            # no explicit thread handling needed here
            pass

async def get_agent_response_async(user_input: str) -> list:
    try:
        raw = await run_agent(user_input)
        data = json.loads(raw.strip().lstrip('```json').rstrip('```'))
        return data if isinstance(data, list) else [data]
    except Exception as e:
        logger.error(f"Error parsing agent response: {e}")
        return []

def get_agent_response(user_input: str) -> list:
    return asyncio.run(get_agent_response_async(user_input))

@st.dialog("Recap", width="large")
def show_validation_dialog():
    validated = st.session_state.get("validated_rows_temp", [])
    st.write(f"Codes sent to **{system_selection}**")
    if validated:
        df = pd.DataFrame([r['code'] for r in validated], columns=["Code"])
        st.table(df)
    else:
        st.warning("No codes validated.")

# Inject minimal CSS for buttons
st.markdown(
    """<style>div.stButton>button{background:#398980;color:#fff;border:none;padding:.5em 1em;border-radius:5px;}</style>""",
    unsafe_allow_html=True
)

# Sidebar & authentication
st.sidebar.image("assets/logo_harmattan.png", width=300)
system_selection = st.sidebar.selectbox(
    "Select a system:",
    ["Hopital Management","DEDALUS","CEGEDIM"]
)
if st.user.is_logged_in:
    with st.sidebar.expander(f"👤 {st.user.name}"):
        st.write(f"**Email:** {st.user.email}")
        st.selectbox("Language", ["Français","English"], index=0, key="user_lang")
        st.button("Logout", on_click=st.logout)
else:
    st.sidebar.button("Login", on_click=st.login)

# Main app
if "agent_response" not in st.session_state:
    st.session_state.agent_response = None
st.image("assets/logo_harmattan.png", width=600)
doctor_notes = st.text_area("Paste doctor's notes here", height=150)

if st.user.is_logged_in:
    if st.button("Send"):
        with st.spinner("Sending request to Harmattan AI...",show_time=True):
            if doctor_notes.strip():
                st.session_state.agent_response = get_agent_response(doctor_notes)
            else:
                st.warning("Enter notes first.")
else:
    st.warning("Login required.")

# Display results and validation
if st.session_state.agent_response:
    df_resp = pd.DataFrame(st.session_state.agent_response)
    for idx, row in df_resp.iterrows():
        cols = st.columns([4.5,4,1.5,1.5,1.5])
        cols[0].markdown(f"**Excerpt**<br>{row['extract']}", unsafe_allow_html=True)
        cols[1].markdown(f"**Description**<br>{row['description']}", unsafe_allow_html=True)
        cols[2].markdown(f"**Code**<br>{row['code']}", unsafe_allow_html=True)
        if row.get('url'): cols[3].markdown(f"[Link]({row['url']})")
        cols[4].checkbox("Validate", key=f"val_{idx}")
        st.markdown("---")
    if st.button("Save"):
        st.session_state.validated_rows_temp = [r.to_dict() for i,r in df_resp.iterrows() if st.session_state.get(f"val_{i}")]
        show_validation_dialog()
