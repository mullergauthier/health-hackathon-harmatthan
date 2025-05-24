import os
import asyncio
import nest_asyncio
import streamlit as st
import json
import pandas as pd
import logging
import re  # Import regex for JSON extraction
from azure.identity.aio import DefaultAzureCredential,ClientSecretCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread, AzureAIAgentSettings

# --- Configuration ---

# Load secrets from Streamlit secrets.toml
# Ensure these keys exist in your secrets.toml:
# [azure]
# AZURE_AI_AGENT_AGENT = "your_agent_id"
# AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = "your_connection_string"
# AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME = "your_deployment_name"
# LOG_LEVEL = "INFO" # Optional: Set to DEBUG for more verbose logs

AGENT_ID = st.secrets.azure.AZURE_AI_AGENT_AGENT
os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"] = st.secrets.azure.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = st.secrets.azure.AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME

# Configure logger (file + console)
LOG_LEVEL_STR = st.secrets.get("azure", {}).get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='debug.log', # Log file
    filemode='w' # Overwrite log file on each run
)
# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# Add console handler to the root logger
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
     root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow running asyncio code within Streamlit's event loop
nest_asyncio.apply()

# --- Azure AI Agent Interaction ---

# Define a timeout for agent calls (in seconds)
AGENT_CALL_TIMEOUT = 120.0 # 2 minutes

async def run_agent(user_input: str) -> str | None:
    """
    Connects to Azure, retrieves the specified agent, and gets a response.

    Args:
        user_input: The text input from the user to send to the agent.

    Returns:
        The raw string response from the agent, or None if an error occurs.
    """
    # Ensure the AGENT_ID corresponds to an agent definition compatible
    # with the expected input/output format of this application.
    logger.info(f"Attempting to run agent {AGENT_ID}...")
    try:
        # Use DefaultAzureCredential, configured via environment variables set from secrets.toml

        creds = ClientSecretCredential(
            tenant_id=st.secrets.azure.AZURE_TENANT_ID,
            client_id=st.secrets.azure.AZURE_CLIENT_ID,
            client_secret=st.secrets.azure.AZURE_CLIENT_SECRET,    
        )

        async with creds, AzureAIAgent.create_client(credential=creds) as client:
            logger.debug("Azure credentials and AI client created.")
            settings = AzureAIAgentSettings.create() # Uses env vars

            # Retrieve the agent definition
            logger.debug(f"Retrieving agent definition for {AGENT_ID}...")
            agent_def = await asyncio.wait_for(
                client.agents.get_agent(agent_id=AGENT_ID),
                timeout=AGENT_CALL_TIMEOUT
            )
            logger.debug("Agent definition retrieved.")

            # Create the agent instance
            agent = AzureAIAgent(client=client, definition=agent_def, settings=settings)
            logger.debug("AzureAIAgent instance created.")

            # Get the agent's response
            # Note: Verify the expected input format for `get_response`.
            # Some agent implementations might expect a list of message dicts,
            # e.g., messages=[{"role": "user", "content": user_input}]
            logger.info("Sending request to agent...")
            response = await asyncio.wait_for(
                agent.get_response(messages=user_input),
                timeout=AGENT_CALL_TIMEOUT
            )
            logger.info("Received response from agent.")
            logger.debug(f"Raw agent response: {response}")
            return str(response)

    except ClientAuthenticationError as e:
        logger.error(f"Azure Authentication Error: {e}", exc_info=True)
        st.error("Authentication failed. Please check Azure credentials configuration.")
        return None
    except HttpResponseError as e:
        logger.error(f"Azure API Error: Status={e.status_code}, Reason={e.reason}, Message={e.message}", exc_info=True)
        st.error(f"An error occurred while communicating with the Azure AI service (Status: {e.status_code}). Please try again later.")
        return None
    except asyncio.TimeoutError:
        logger.error(f"Agent call timed out after {AGENT_CALL_TIMEOUT} seconds.")
        st.error("The request to the AI agent timed out. Please try again.")
        return None
    except Exception as e:
        # Catch unexpected errors during agent interaction
        logger.error(f"Unexpected error during agent interaction: {e}", exc_info=True)
        st.error("An unexpected error occurred while processing your request. Please check the logs.")
        return None

def extract_json_from_string(text: str) -> str | None:
    """
    Extracts JSON content potentially wrapped in markdown code fences.

    Args:
        text: The string potentially containing JSON.

    Returns:
        The extracted JSON string, or None if no JSON object/array is found.
    """
    # Regex to find JSON object '{...}' or array '[...]' potentially wrapped in ```json ... ```
    # It handles potential leading/trailing whitespace and the markdown fences.
    match = re.search(r'```(?:json)?\s*([\[\{].*[\]\}])\s*```|([\[\{].*[\]\}])', text, re.DOTALL | re.IGNORECASE)
    if match:
        # Return the first non-None captured group
        return match.group(1) if match.group(1) else match.group(2)
    logger.warning("Could not find JSON object or array in the agent's response.")
    return None

async def get_agent_response_async(user_input: str) -> list | None:
    """
    Runs the agent, parses the JSON response, and handles errors.

    Args:
        user_input: The text input from the user.

    Returns:
        A list containing the parsed data from the agent, or None if an error occurs.
    """
    raw_response = await run_agent(user_input)
    if raw_response is None:
        return None # Error already handled and logged in run_agent

    json_string = extract_json_from_string(raw_response)
    if json_string is None:
        st.error("Could not extract valid JSON data from the agent's response.")
        logger.error(f"Failed to extract JSON from raw response: {raw_response}")
        return None

    try:
        logger.debug(f"Attempting to parse JSON: {json_string}")
        data = json.loads(json_string)
        # Ensure the result is always a list
        if isinstance(data, list):
            logger.info(f"Successfully parsed agent response into a list of {len(data)} items.")
            return data
        elif isinstance(data, dict):
             logger.info("Successfully parsed agent response into a single dictionary, wrapping in a list.")
             return [data]
        else:
            logger.error(f"Parsed JSON is not a list or dictionary, type: {type(data)}")
            st.error("The agent returned data in an unexpected format.")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}", exc_info=True)
        logger.error(f"Problematic JSON string: {json_string}")
        st.error("Failed to parse the JSON data received from the agent.")
        return None
    except Exception as e:
        # Catch unexpected errors during parsing/processing
        logger.error(f"Unexpected error processing agent response: {e}", exc_info=True)
        st.error("An unexpected error occurred while processing the agent's response.")
        return None

def get_agent_response_sync(user_input: str) -> list | None:
    """Synchronous wrapper for the async agent call."""
    # This runs the async function in the current event loop (managed by nest_asyncio)
    return asyncio.run(get_agent_response_async(user_input))

# --- Streamlit UI ---

@st.dialog("Recap", width="large")
def show_validation_dialog(validated_data: list[dict], system_name: str):
    """
    Displays the validation recap dialog.

    Args:
        validated_data: A list of dictionaries, where each dict represents a validated row.
        system_name: The name of the system selected in the sidebar.
    """
    st.write(f"Codes selected for sending to **{system_name}**:")
    if validated_data:
        # Extract only the 'code' field for display in the table
        codes_to_display = [{'Code': row.get('code', 'N/A')} for row in validated_data]
        df = pd.DataFrame(codes_to_display)
        st.table(df)
        # You might want to add logic here to actually *send* the data
        st.success("Data ready for transmission (implementation pending).")
        logger.info(f"{len(validated_data)} codes validated for system {system_name}.")
    else:
        st.warning("No codes were selected for validation.")
        logger.info(f"Validation dialog shown for system {system_name}, but no codes were selected.")

# Inject minimal CSS for button styling
# For more complex styling, consider using st.set_page_config(layout="wide")
# and external CSS files or Streamlit Theming.
st.markdown(
    """<style>
       div.stButton > button {
           background-color: #398980; /* Example color */
           color: #ffffff;
           border: none;
           padding: 0.5em 1em;
           border-radius: 5px;
           font-weight: bold;
           cursor: pointer; /* Add pointer cursor */
       }
       div.stButton > button:hover {
           background-color: #2a6a60; /* Darker shade on hover */
       }
       /* Style the logout button differently if needed */
       /* You might need to inspect the HTML to find a more specific selector */
    </style>""",
    unsafe_allow_html=True
)

# --- Sidebar ---
st.sidebar.image("assets/logo_harmattan.png", width=250) # Adjust width as needed
system_selection = st.sidebar.selectbox(
    "Select Target System:",
    ["Hopital Management", "DEDALUS", "CEGEDIM"],
    key="system_selection" # Add key for stability
)

# Authentication check and user info in sidebar
if st.user.is_logged_in:
    with st.sidebar.expander(f"👤 User: {st.user.name}", expanded=False):
        st.write(f"**Email:** {st.user.email}")
        # Example: Language selection (doesn't do anything in this code)
        st.selectbox("Language", ["Français", "English"], index=0, key="user_lang")
        if st.button("Logout", key="logout_button"):
             st.logout() # Use Streamlit's built-in logout
else:
    st.sidebar.button("Login", on_click=st.login("auth0"), key="login_button") # Use Streamlit's built-in login

# --- Main Application Area ---

# Initialize session state for agent response if it doesn't exist
if "agent_response_data" not in st.session_state:
    st.session_state.agent_response_data = None # Store the parsed list here
if "validation_states" not in st.session_state:
    st.session_state.validation_states = {} # Store checkbox states {index: bool}

st.image("assets/logo_harmattan.png", width=500) # Adjust width as needed
st.header("AI Medical Code Assistant")
st.markdown("Paste the doctor's notes below and click 'Analyze Notes' to get suggested codes.")

doctor_notes = st.text_area("Doctor's Notes:", height=200, key="doctor_notes_input", placeholder="Enter clinical notes here...")

# Main action button - only enabled if logged in
if st.user.is_logged_in:
    if st.button("Analyze Notes", key="analyze_button", type="primary"):
        if doctor_notes.strip():
            with st.spinner("Sending request to Harmattan AI... Please wait."): # Removed show_time
                logger.info("Analyze button clicked, processing notes.")
                # Call the synchronous wrapper which handles the async call
                response_data = get_agent_response_sync(doctor_notes)
                st.session_state.agent_response_data = response_data # Store results or None
                # Reset validation states when new results are fetched
                st.session_state.validation_states = {}
                if response_data is None:
                    # Error message already shown by get_agent_response_async/run_agent
                    logger.warning("Agent analysis resulted in an error or no data.")
                elif not response_data:
                     st.info("The analysis did not return any specific codes for the provided notes.")
                     logger.info("Agent analysis completed but returned an empty list.")
                else:
                     st.success(f"Analysis complete. Found {len(response_data)} potential codes.")
                     logger.info(f"Agent analysis successful, {len(response_data)} items returned.")

        else:
            st.warning("Please paste the doctor's notes into the text area before analyzing.")
            logger.warning("Analyze button clicked, but input notes were empty.")
else:
    st.warning("Please log in using the sidebar to use the analysis feature.")

# --- Display Results and Validation ---
st.markdown("---") # Visual separator
st.subheader("Analysis Results")

if st.session_state.agent_response_data:
    results = st.session_state.agent_response_data
    df_resp = pd.DataFrame(results) # Create DataFrame for easier handling

    # Use st.form to group checkboxes and the save button
    with st.form(key="validation_form"):
        # Display results in columns
        # Headers outside the loop
        cols_header = st.columns([4, 3, 2, 1, 1]) # Adjusted widths
        cols_header[0].markdown("**Excerpt**")
        cols_header[1].markdown("**Description**")
        cols_header[2].markdown("**Code**")
        cols_header[3].markdown("**Link**")
        cols_header[4].markdown("**Check**")
        st.markdown("---") # Separator after headers

        for idx, row in df_resp.iterrows():
            cols = st.columns([4.5,4,1.5,1.5,1.5]) # Match header widths

            # Use st.text or st.write for safety - avoids unsafe HTML
            cols[0].text(row.get('extract', 'N/A')) # Display raw text
            cols[1].text(row.get('description', 'N/A'))
            cols[2].text(row.get('code', 'N/A'))

            # Display link safely if present
            url = row.get('url')
            if url:
                # Basic validation: ensure it looks like a URL
                if isinstance(url, str) and url.startswith(('http://', 'https://')):
                    cols[3].link_button("Link", url, help=f"Open link for code {row.get('code', '')}")
                else:
                    cols[3].text("-") # Placeholder if URL is invalid
                    logger.warning(f"Row {idx} has an invalid URL: {url}")
            else:
                cols[3].text("-") # Placeholder if no URL

            # Checkbox for validation - state managed by Streamlit using the key
            # Use the index as part of the key for uniqueness
            validation_key = f"validate_{idx}"
            is_validated = cols[4].checkbox("Validate", key=validation_key, value=st.session_state.validation_states.get(validation_key, False),label_visibility="collapsed")
            # Update the central validation state tracker immediately (though form submission is the final trigger)
            st.session_state.validation_states[validation_key] = is_validated

            st.markdown("---") # Separator between rows

        # Submit button for the form
        submitted = st.form_submit_button("Save Validated Codes")
        if submitted:
            logger.info("Save Validated Codes button clicked.")
            # Collect validated rows based on the checkbox states within the form's submission context
            validated_rows_data = []
            for i, r in df_resp.iterrows():
                 if st.session_state.get(f"validate_{i}", False): # Check the state at submission time
                     validated_rows_data.append(r.to_dict()) # Convert row to dict

            logger.debug(f"Data prepared for validation dialog: {validated_rows_data}")
            # Call the dialog function, passing the validated data and system name
            show_validation_dialog(validated_data=validated_rows_data, system_name=system_selection)

elif st.session_state.agent_response_data is None and st.user.is_logged_in:
     # Only show this if logged in and no analysis has been run yet or failed
     if 'analyze_button' in st.session_state and st.session_state.analyze_button:
         # If analysis was attempted but failed (response is None), error shown elsewhere
         pass
     else:
        st.info("Enter doctor's notes and click 'Analyze Notes' to begin.")

# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.info("Harmattan AI Assistant v1.1") # Example version info

