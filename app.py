import asyncio
import nest_asyncio
import streamlit as st
import json
import pandas as pd
import logging
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread

# Configuration du logger pour VS et pour un fichier de log
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log',  # Les logs seront enregistrés dans ce fichier
    filemode='w'           # 'w' pour écraser à chaque lancement, 'a' pour ajouter
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.debug("Initialisation du script Streamlit avec intégration de logging.")

# Permet de faire tourner des boucles asyncio dans Streamlit
nest_asyncio.apply()

# Nettoie la réponse JSON retournée sous forme de chaîne (retire les balises Markdown)
def clean_json_response(response):
    logger.debug("Nettoyage de la réponse JSON reçue.")
    if isinstance(response, str):
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):].lstrip()
        if response.endswith("```"):
            response = response[:-len("```")].rstrip()
    return response

# Fonction asynchrone pour interroger l'agent Azure OpenAI
async def run_agent(user_input: str) -> str:
    logger.debug("Démarrage de la requête asynchrone vers l'agent Azure.")
    async with DefaultAzureCredential() as creds, AzureAIAgent.create_client(credential=creds) as client:
        # Récupération de la définition de l'agent depuis Azure
        agent_definition = await client.agents.get_agent(agent_id="asst_E5nFroutEcRYyKkXsLkMwPvJ")
        logger.debug("Définition de l'agent récupérée.")
        agent = AzureAIAgent(client=client, definition=agent_definition)
        thread: AzureAIAgentThread = None
        try:
            # Envoie le message et récupération de la réponse de l'agent
            logger.debug(f"Envoi de la requête utilisateur : {user_input}")
            response = await agent.get_response(messages=user_input, thread=thread)
            logger.debug(f"Réponse reçue de l'agent : {response}")
            return str(response)
        finally:
            if thread:
                await thread.delete()
                logger.debug("Thread de l'agent supprimé.")

# Fonction de fallback utilisant les données fournies
def get_agent_response_fallback(notes: str):
    logger.debug("Utilisation de la réponse de secours (fallback).")
    return [
        {
            "extract": "rééducation et réautonomisation chez une patiente de 83 ans dans les suites d'une osteosynthèse sur fracture periprothétique du femur proximal gauche avec mise en place d'un cerclage sur PTH gauche",
            "code": "S72.00",
            "description": "Fracture du col du fémur",
            "url": "https://icd.who.int/browse10/2019/en#/S72.00"
        },
        {
            "extract": "prise en charge chirurgicale d'un HSD chronique",
            "code": "I62.0",
            "description": "Hématome sous-dural aigu (traumatique)",
            "url": "https://icd.who.int/browse10/2019/en#/I62.0"
        },
        {
            "extract": "HTA",
            "code": "I10",
            "description": "Hypertension essentielle (primaire)",
            "url": "https://icd.who.int/browse10/2019/en#/I10"
        },
        {
            "extract": "trouble du sommeil",
            "code": "F51.9",
            "description": "Trouble du sommeil, sans précision",
            "url": "https://icd.who.int/browse10/2019/en#/F51.9"
        },
        {
            "extract": "trouble de la marche",
            "code": "R26.9",
            "description": "Trouble de la marche, sans précision",
            "url": "https://icd.who.int/browse10/2019/en#/R26.9"
        },
        {
            "extract": "perte d'autonomie",
            "code": "R53",
            "description": "Malaise et fatigue",
            "url": "https://icd.who.int/browse10/2019/en#/R53"
        },
        {
            "extract": "traitement anticoagulant préventif",
            "code": "Z79.01",
            "description": "Utilisation à long terme (actuelle) d'anticoagulants",
            "url": "https://icd.who.int/browse10/2019/en#/Z79.01"
        },
        {
            "extract": "fibrillation atriale emboligène (AVC ischémique)",
            "code": "I48.0",
            "description": "Fibrillation auriculaire",
            "url": "https://icd.who.int/browse10/2019/en#/I48.0"
        },
        {
            "extract": "cardiopathie hypokinétique",
            "code": "I50.1",
            "description": "Insuffisance ventriculaire gauche",
            "url": "https://icd.who.int/browse10/2019/en#/I50.1"
        },
        {
            "extract": "VPPB",
            "code": "H81.1",
            "description": "Vertige paroxystique bénin",
            "url": "https://icd.who.int/browse10/2019/en#/H81.1"
        }
    ]

# Fonction synchrone qui enveloppe l'appel à l'agent et gère un plan de secours en cas de problème
def get_agent_response(user_input: str) -> list:
    logger.debug("Appel à get_agent_response avec input utilisateur.")
    try:
        # Timeout de 30 secondes pour éviter un temps de réponse trop long
        raw_response = asyncio.run(asyncio.wait_for(run_agent(user_input), timeout=30))
        raw_response = clean_json_response(raw_response)
        if not raw_response:
            logger.warning("Aucune réponse reçue, utilisation du fallback.")
            return get_agent_response_fallback(user_input)
        try:
            json_response = json.loads(raw_response)
            if isinstance(json_response, list):
                logger.debug("Réponse JSON décodée correctement.")
                return json_response
            elif isinstance(json_response, dict):
                logger.warning("Réponse JSON est un objet unique, encapsulation dans une liste.")
                return [json_response]
            else:
                logger.warning("Réponse JSON inattendue, utilisation du fallback.")
                return get_agent_response_fallback(user_input)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON : {e}")
            return get_agent_response_fallback(user_input)
    except Exception as e:
        logger.error(f"Exception dans get_agent_response : {e}")
        return get_agent_response_fallback(user_input)

# Fenêtre de dialogue Streamlit affichant les codes validés
@st.dialog("Recapitulatif", width="large")
def show_validation_dialog():
    validated_rows = st.session_state.get("validated_rows_temp", [])
    st.write(f"Codes envoyés à l'application **{system_selection}**")
    if validated_rows:
        st.markdown("### Liste récapitulative des codes validés")
        validated_codes = [row["code"] for row in validated_rows]
        df_codes = pd.DataFrame(validated_codes, columns=["Code Validé"])
        st.table(df_codes)
    else:
        st.warning("Aucun code n'a été validé.")

# Inject custom CSS
st.markdown("""
<style>
/* Apply custom styles to all Streamlit buttons */
div.stButton > button {
    background-color: #398980;  /* Primary color */
    color: #FFFFFF;             /* White text */
    border: none;
    padding: 0.5em 1em;
    border-radius: 5px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: url("assets\Screenshot_20250329_224946_Drive.png") no-repeat center center;
        background-size: 60%;
        opacity: 0.55; /* transparence du logo */
        z-index: -1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("assets/Screenshot_20250329_224825_Drive.png", width=300)
system_selection = st.sidebar.selectbox(
    "Sélectionnez un système :",
    ["Hopital Management", "DEDALUS", "CEGEDIM"]
)

# Initialisation du session_state si non présent
if "agent_response" not in st.session_state:
    st.session_state.agent_response = None

# Titre principal et champ de saisie pour les notes du médecin
st.image("assets/Screenshot_20250329_224825_Drive.png", width=500)
doctor_notes = st.text_area("Copier/coller vos notes ici", height=150)

# Bouton d'envoi des notes au modèle IA
if st.button("Envoyer"):
    if doctor_notes.strip():
        logger.debug("Notes du médecin soumises, envoi à l'agent.")
        with st.spinner("Envoi de la demande à l'agent Harmattan...", show_time=True):
            st.session_state.agent_response = get_agent_response(doctor_notes)
    else:
        st.warning("Veuillez saisir les notes du médecin.")
        logger.warning("Aucune note saisie par l'utilisateur.")

# Affichage des résultats si une réponse est disponible
if st.session_state.agent_response:
    st.subheader("Liste des codes ICD-10, veuillez les verifier")
    df_response = pd.DataFrame(st.session_state.agent_response)
    logger.debug("Affichage de la réponse de l'agent dans Streamlit.")

    # Affichage de chaque ligne de résultat dans une grille de colonnes
    for idx, row in df_response.iterrows():
        col1, col2, col3, col4, col5 = st.columns([4.5, 4, 1.5, 1.5, 1.5])
        with col1:
            st.markdown(f"**Extrait**<br>{row['extract']}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Description**<br>{row['description']}", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**Code**<br>{row['code']}", unsafe_allow_html=True)
        with col4:
            if row.get("url"):
                st.markdown(f"[Lien]({row['url']})")
        with col5:
            checkbox_key = f"validation_{idx}"
            st.checkbox("Check", key=checkbox_key)
        st.markdown("---")

    # Bouton de sauvegarde des validations cochées
    if st.button("Sauvegarder les validations"):
        validated_rows = [row.to_dict() for idx, row in df_response.iterrows()
                          if st.session_state.get(f"validation_{idx}", False)]
        st.session_state.validated_rows_temp = validated_rows
        logger.debug(f"Codes validés sauvegardés : {validated_rows}")
        show_validation_dialog()
