import os
import asyncio
import nest_asyncio
import streamlit as st
import json
import pandas as pd
import logging
import re  # Import regex for JSON extraction
from azure.identity.aio import ClientSecretCredential # DefaultAzureCredential removed as ClientSecretCredential is used directly
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings # AzureAIAgentThread removed as it's not used
from azure.identity import ClientSecretCredential as SyncClientSecretCredential
from azure.ai.projects import AIProjectClient
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# --- Configuration ---

# Load secrets from Streamlit secrets.toml
# Ensure these keys exist in your secrets.toml:
# [azure]
# AZURE_AI_AGENT_AGENT = "your_agent_id"
# AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = "your_connection_string"
# AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME = "your_deployment_name"
# AZURE_TENANT_ID = "your_tenant_id"
# AZURE_CLIENT_ID = "your_client_id"
# AZURE_CLIENT_SECRET = "your_client_secret"
# LOG_LEVEL = "INFO" # Optional: Set to DEBUG for more verbose logs

AGENT_ID = st.secrets.azure.AZURE_AI_AGENT_AGENT
os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"] = st.secrets.azure.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = st.secrets.azure.AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME

import logging

# — NEW LOGGING CONFIG — 

# 1) Configure the **root** logger to only record INFO+ (so DEBUG/VERBOSE is dropped immediately):
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="debug.log",
    filemode="w",
)

# 2) Now create your module‐level logger so you can still call `logger.debug(...)` if you want:
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 3) Quiet down other noisy libraries at WARNING+ only:
for lib in ("azure", "opentelemetry", "httpx", "urllib3"):
    logging.getLogger(lib).setLevel(logging.ERROR)

# 4) (Optional) Re-attach a console handler for INFO+ to stdout:
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)



# --- Azure AI Foundry Telemetry Initialization ---
try:
    endpoint       = st.secrets.azure.AZURE_AI_PROJECT_ENDPOINT
    subscription   = st.secrets.azure.AZURE_SUBSCRIPTION_ID
    resource_group = st.secrets.azure.AZURE_RESOURCE_GROUP_NAME
    project_name   = st.secrets.azure.AZURE_AI_PROJECT_NAME
    tenant_id      = st.secrets.azure.AZURE_TENANT_ID
    client_id      = st.secrets.azure.AZURE_CLIENT_ID
    client_secret  = st.secrets.azure.AZURE_CLIENT_SECRET

    # Sauter uniquement si les secrets essentiels manquent
    if not all([endpoint, subscription, resource_group, project_name, tenant_id, client_id, client_secret]):
        logger.warning("⚠️ Secrets Azure Foundry incomplets ; télémétrie désactivée.")
    else:
        creds_sync = SyncClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
        project_client = AIProjectClient(
            endpoint=endpoint,
            subscription_id=subscription,
            resource_group_name=resource_group,
            project_name=project_name,
            credential=creds_sync
        )
        try:
            ai_conn_str = project_client.telemetry.get_connection_string()
            if ai_conn_str:
                os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"
                configure_azure_monitor(connection_string=ai_conn_str)
                logger.info("✅ Télémétrie Azure AI Foundry configurée.")
            else:
                logger.warning("⚠️ Pas de ressource App Insights liée ; télémétrie désactivée.")
        except ClientAuthenticationError:
            logger.warning("⚠️ Permissions insuffisantes pour la télémétrie Foundry ; ignore.")
except Exception as ex:
    logger.error(f"Erreur d’initialisation de la télémétrie : {ex}", exc_info=True)


tracer = trace.get_tracer("HarmattanAI")


# Apply nest_asyncio to allow running asyncio code wi"thin Streamlit's event loop
nest_asyncio.apply()

# --- Azure AI Agent Interaction ---

# Define a timeout for agent calls (in seconds)
AGENT_CALL_TIMEOUT = 120.0  # 2 minutes

async def run_agent(user_input: str) -> str | None:
    """
    Connects to Azure, retrieves the specified agent, and gets a response.
    Telemetry is captured for the agent interaction.

    Args:
        user_input: The text input from the user to send to the agent.

    Returns:
        The raw string response from the agent, or None if an error occurs.
    """
    # Ensure the AGENT_ID corresponds to an agent definition compatible
    # with the expected input/output format of this application.
    with tracer.start_as_current_span("AI-Agent.run_agent") as span: # Changed '/' to '.' for potential naming convention consistency
        span.set_attribute("ai.agent_id", AGENT_ID)
        span.set_attribute("ai.timeout_seconds", AGENT_CALL_TIMEOUT)
        span.set_attribute("ai.user_input_preview", user_input[:100]) # Add preview of input for context

        logger.info(f"Attempting to run agent {AGENT_ID}...")
        try:
            # Ensure all necessary secrets for ClientSecretCredential are present
            if not all([st.secrets.azure.get("AZURE_TENANT_ID"),
                        st.secrets.azure.get("AZURE_CLIENT_ID"),
                        st.secrets.azure.get("AZURE_CLIENT_SECRET")]):
                logger.error("Missing Azure credentials (TENANT_ID, CLIENT_ID, or CLIENT_SECRET) for AzureAIAgent.")
                st.error("Azure credentials for the AI Agent are not fully configured. Please check secrets.")
                span.set_status(Status(StatusCode.ERROR, "Missing Azure credentials for AI Agent"))
                return None

            creds_async = ClientSecretCredential(
                tenant_id=st.secrets.azure.AZURE_TENANT_ID,
                client_id=st.secrets.azure.AZURE_CLIENT_ID,
                client_secret=st.secrets.azure.AZURE_CLIENT_SECRET,
            )

            async with creds_async, AzureAIAgent.create_client(credential=creds_async) as client:
                logger.debug("Azure credentials and AI client created.")
                settings = AzureAIAgentSettings.create()  # Uses env vars

                # Retrieve the agent definition
                logger.debug(f"Retrieving agent definition for {AGENT_ID}...")
                agent_def = await asyncio.wait_for(
                    client.agents.get_agent(agent_id=AGENT_ID),
                    timeout=AGENT_CALL_TIMEOUT
                )
                logger.debug("Agent definition retrieved.")
                span.add_event("agent.definition.retrieved")

                # Create the agent instance
                agent = AzureAIAgent(client=client, definition=agent_def, settings=settings)
                logger.debug("AzureAIAgent instance created.")

                # Get the agent's response
                logger.info("Sending request to agent...")
                response_content = None # Initialize response_content

                # *** TELEMETRY FIX: Corrected duplicated call and added specific span for get_response ***
                with tracer.start_as_current_span("AI-Agent.get_response") as rsp_span:
                    rsp_span.set_attribute("ai.user_input_length", len(user_input))
                    try:
                        # Note: Verify the expected input format for `get_response`.
                        # Some agent implementations might expect a list of message dicts,
                        # e.g., messages=[{"role": "user", "content": user_input}]
                        # Assuming user_input is the correct format for your agent.
                        api_response = await asyncio.wait_for(
                            agent.get_response(messages=user_input), # Single call to agent
                            timeout=AGENT_CALL_TIMEOUT
                        )
                        response_content = str(api_response) # Convert to string
                        rsp_span.set_attribute("ai.response_preview", api_response) # Add preview of response
                        rsp_span.add_event("agent.get_response.succeeded")
                        # Span status is OK by default if no exception
                    except Exception as e_inner:
                        logger.error(f"Error during agent.get_response: {e_inner}", exc_info=True)
                        rsp_span.record_exception(e_inner)
                        rsp_span.set_status(Status(StatusCode.ERROR, f"Agent get_response failed: {type(e_inner).__name__}"))
                        raise # Re-raise to be caught by the outer try-except, which will mark the parent span

                span.add_event("agent.response.processed") # Event for the outer span
                logger.info("Received response from agent.")
                logger.debug(f"Raw agent response: {response_content}")
                # Set status OK for the outer span if we reached here successfully
                span.set_status(Status(StatusCode.OK))
                return response_content

        except ClientAuthenticationError as e:
            logger.error(f"Azure Authentication Error: {e}", exc_info=True)
            st.error("Authentication failed. Please check Azure credentials configuration.")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, "Azure Authentication Error"))
            return None
        except HttpResponseError as e:
            logger.error(f"Azure API Error: Status={e.status_code}, Reason={e.reason}, Message={e.message}", exc_info=True)
            st.error(f"An error occurred while communicating with the Azure AI service (Status: {e.status_code}). Please try again later.")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, f"Azure API Error: {e.status_code}"))
            span.set_attribute("http.status_code", e.status_code) # Add http status code if available
            return None
        except asyncio.TimeoutError:
            logger.error(f"Agent call timed out after {AGENT_CALL_TIMEOUT} seconds.")
            st.error("The request to the AI agent timed out. Please try again.")
            # TimeoutError is an Exception, so record_exception will work.
            # Create a TimeoutError instance to pass to record_exception if not automatically available.
            timeout_exc = asyncio.TimeoutError(f"Agent call timed out after {AGENT_CALL_TIMEOUT} seconds.")
            span.record_exception(timeout_exc)
            span.set_status(Status(StatusCode.ERROR, "Agent call timed out"))
            return None
        except Exception as e:
            logger.error(f"Unexpected error during agent interaction: {e}", exc_info=True)
            # Show the actual exception text in the UI
            st.error(f"An unexpected error occurred: {e}")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, f"Unexpected error: {type(e).__name__}"))
            return None


def extract_json_from_string(text: str) -> str | None:
    """
    Extracts JSON content potentially wrapped in markdown code fences.

    Args:
        text: The string potentially containing JSON.

    Returns:
        The extracted JSON string, or None if no JSON object/array is found.
    """
    if not text: # Handle empty or None input
        logger.warning("Input text for JSON extraction is empty or None.")
        return None
    # Regex to find JSON object '{...}' or array '[...]' potentially wrapped in ```json ... ```
    # It handles potential leading/trailing whitespace and the markdown fences.
    match = re.search(r'```(?:json)?\s*([\[\{].*[\]\}])\s*```|([\[\{].*[\]\}])', text, re.DOTALL | re.IGNORECASE)
    if match:
        # Return the first non-None captured group
        json_str = match.group(1) if match.group(1) else match.group(2)
        logger.debug(f"Extracted JSON string: {json_str[:200]}...") # Log preview
        return json_str
    logger.warning("Could not find JSON object or array in the agent's response.")
    logger.debug(f"Full text searched for JSON: {text[:500]}...") # Log preview of text that failed extraction
    return None

async def get_agent_response_async(user_input: str) -> list | None:
    """
    Runs the agent, parses the JSON response, and handles errors.
    Telemetry for the raw agent call is handled within run_agent.
    This function focuses on parsing and could have its own span if complex.

    Args:
        user_input: The text input from the user.

    Returns:
        A list containing the parsed data from the agent, or None if an error occurs.
    """
    # Consider adding a span here if parsing logic is complex or error-prone
    with tracer.start_as_current_span("App.parse_agent_response") as parse_span:
        raw_response = await run_agent(user_input)
        if raw_response is None:
            parse_span.set_status(Status(StatusCode.ERROR, "Agent returned no response")) # If span added
            return None  # Error already handled and logged in run_agent

    json_string = extract_json_from_string(raw_response)
    if json_string is None:
        st.error("Could not extract valid JSON data from the agent's response.")
        logger.error(f"Failed to extract JSON from raw response: {raw_response}")
        parse_span.set_status(Status(StatusCode.ERROR, "JSON extraction failed")) # If span added
        parse_span.set_attribute("app.raw_response_preview", raw_response[:200]) # If span added
        return None

    try:
        logger.debug(f"Attempting to parse JSON: {json_string[:200]}...")
        data = json.loads(json_string)
        # Ensure the result is always a list
        if isinstance(data, list):
            logger.info(f"Successfully parsed agent response into a list of {len(data)} items.")
            parse_span.set_attribute("app.parsed_item_count", len(data)) # If span added
            return data
        elif isinstance(data, dict):
            logger.info("Successfully parsed agent response into a single dictionary, wrapping in a list.")
            parse_span.set_attribute("app.parsed_item_count", 1) # If span added
            return [data]
        else:
            logger.error(f"Parsed JSON is not a list or dictionary, type: {type(data)}")
            st.error("The agent returned data in an unexpected format.")
            parse_span.set_status(Status(StatusCode.ERROR, f"Unexpected JSON type: {type(data)}")) # If span added
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}", exc_info=True)
        logger.error(f"Problematic JSON string: {json_string}")
        st.error("Failed to parse the JSON data received from the agent.")
        # if 'parse_span' in locals(): # If span added
        #     parse_span.record_exception(e)
        #     parse_span.set_status(Status(StatusCode.ERROR, "JSONDecodeError"))
        return None
    except Exception as e:
        # Catch unexpected errors during parsing/processing
        logger.error(f"Unexpected error processing agent response: {e}", exc_info=True)
        st.error("An unexpected error occurred while processing the agent's response.")
        # if 'parse_span' in locals(): # If span added
        #     parse_span.record_exception(e)
        #     parse_span.set_status(Status(StatusCode.ERROR, f"Unexpected parsing error: {type(e).__name__}"))
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
    </style>""",
    unsafe_allow_html=True
)

# --- Sidebar ---
# Ensure asset path is correct or handle potential FileNotFoundError
try:
    st.sidebar.image("assets/logo_harmattan.png", width=250) # Adjust width as needed
except FileNotFoundError:
    st.sidebar.warning("Logo image not found at assets/logo_harmattan.png")
except Exception as e: # Catch other potential errors from st.image, like UnidentifiedImageError
    st.sidebar.warning(f"Could not load logo: {e}")


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

# Ensure asset path is correct or handle potential FileNotFoundError
try:
    st.image("assets/logo_harmattan.png", width=500) # Adjust width as needed
except FileNotFoundError:
    st.warning("Main logo image not found at assets/logo_harmattan.png")
except Exception as e:
    st.warning(f"Could not load main logo: {e}")

st.header("AI Medical Code Assistant")
st.markdown("Paste the doctor's notes below and click 'Analyze Notes' to get suggested codes.")

doctor_notes = st.text_area("Doctor's Notes:", height=200, key="doctor_notes_input", placeholder="Enter clinical notes here...")

# Main action button - only enabled if logged in
if st.user.is_logged_in:
    if st.button("Analyze Notes", key="analyze_button", type="primary"):
        if doctor_notes.strip():
            with st.spinner("Sending request to Harmattan AI... Please wait.",show_time=True):
                logger.info(f"Analyze button clicked, processing notes for system: {system_selection}.") # Added system_selection to log
                # Call the synchronous wrapper which handles the async call
                response_data = get_agent_response_sync(doctor_notes)
                st.session_state.agent_response_data = response_data # Store results or None
                # Reset validation states when new results are fetched
                st.session_state.validation_states = {}
                if response_data is None:
                    logger.warning("Agent analysis resulted in an error or no data.")
                    # Error message already shown by get_agent_response_async/run_agent
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
    # Ensure results is a list of dicts for DataFrame compatibility
    if not all(isinstance(item, dict) for item in results):
        st.error("Agent response data is not in the expected format (list of dictionaries).")
        logger.error(f"Agent response data malformed: {results}")
    else:
        df_resp = pd.DataFrame(results)

        with st.form(key="validation_form"):
            # Headers outside the loop
            # Adjusted column widths for better layout
            cols_header = st.columns([4, 3, 2, 1, 1]) # Excerpt, Description, Code, Link, Check
            cols_header[0].markdown("**Excerpt**")
            cols_header[1].markdown("**Description**")
            cols_header[2].markdown("**Code**")
            cols_header[3].markdown("**Link**")
            cols_header[4].markdown("**Validate**") # Changed from "Check" to "Validate" for clarity
            st.markdown("---") # Separator after headers

            for idx, row in df_resp.iterrows():
                # Match header column widths
                cols = st.columns([4.5,4,1.5,1.5,1.5])

                cols[0].text(row.get('extract', 'N/A'))
                cols[1].text(row.get('description', 'N/A'))
                cols[2].text(row.get('code', 'N/A'))

                url = row.get('url')
                if url and isinstance(url, str) and (url.startswith(('http://', 'https://')) or url.startswith('www.')):
                    # Ensure URL is absolute if it starts with www.
                    display_url = f"https://{url}" if url.startswith('www.') else url
                    cols[3].link_button("Link", display_url, help=f"Open link for code {row.get('code', '')}")
                else:
                    cols[3].text("-")
                    if url: # Log if a URL was present but invalid
                        logger.warning(f"Row {idx} has an invalid or missing URL: '{url}'")

                validation_key = f"validate_{idx}"
                # Default to False if key not in validation_states
                current_validation_state = st.session_state.validation_states.get(validation_key, False)
                is_validated = cols[4].checkbox(
                    "", # Removed label as header "Validate" serves this purpose
                    key=validation_key,
                    value=current_validation_state,
                    label_visibility="hidden"
                )
                st.session_state.validation_states[validation_key] = is_validated
                st.markdown("---")

            submitted = st.form_submit_button("Save Validated Codes")
            if submitted:
                logger.info("Save Validated Codes button clicked.")
                validated_rows_data = []
                for i, r_series in df_resp.iterrows(): # r_series is a pandas Series
                    # Use the persisted checkbox state from st.session_state.validation_states
                    if st.session_state.validation_states.get(f"validate_{i}", False):
                        validated_rows_data.append(r_series.to_dict())

                logger.debug(f"Data prepared for validation dialog: {validated_rows_data}")
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
st.sidebar.info("Harmattan AI Assistant v1.2") 