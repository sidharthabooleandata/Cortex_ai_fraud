import streamlit as st
import snowflake.connector
from cryptography.hazmat.primitives import serialization

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Boolean Chat_new_2", layout="wide")

# ------------------------
# SIDEBAR STYLING
# ------------------------
st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh; 
        display: flex;
        flex-direction: column;
    }
    div[data-testid="stChatMessage"] > div:nth-child(1) {
        display: none !important;
    }
    div[data-testid="stChatMessage"] > div:nth-child(2) {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    .sidebar-top {flex: 0 0 auto; text-align: center; padding: 10px 0;}
    .sidebar-middle {
        flex: 1 0 auto; display: flex; justify-content: center;
        align-items: center; text-align: center; padding: 50px;
    }
    .sidebar-bottom {
        flex: 0 0 auto; text-align: center; padding: 30px 30px;
        display: flex; justify-content: center; gap: 15px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div class="sidebar-top">
            <img src="https://booleandata.com/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1-980x316.png" style="max-width:100%;">
        </div>
        <div class="sidebar-middle">
            <div>
                <h5>🚀 About Us</h5>
                <p>We are a data-driven company revolutionizing the insurance industry 
                through predictive analytics. Our models help detect fraudulent claims 
                with high accuracy and transparency.</p>
            </div>
        </div>
        <div class="sidebar-bottom">
            <a href="https://booleandata.ai/" target="_blank">🌐</a>
            <a href="https://www.facebook.com/Booleandata" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24">
            </a>
            <a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24">
            </a>
            <a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24">
            </a>
        </div>
    """, unsafe_allow_html=True)

# ------------------------
# SNOWFLAKE CONNECTION (cached)
# ------------------------
# SNOWFLAKE CONNECTION (cached)
# ------------------------
@st.cache_resource(show_spinner=False)
def get_connection():
    private_key_bytes = st.secrets["snowflake"]["private_key"].encode()

    private_key = serialization.load_pem_private_key(
        private_key_bytes,
        password=None,   # Or b"your_passphrase" if your key has one
    )

    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    return snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],   # e.g. "DIC19309.us-east-1"
        user=st.secrets["snowflake"]["user"],
        role=st.secrets["snowflake"]["role"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        private_key=pkb,
    )

conn = get_connection()
cursor = conn.cursor()

# ------------------------
# HELPERS
# ------------------------
@st.cache_data(show_spinner=False)
def get_embedding(text: str):
    """Cached embeddings for speed."""
    query = "SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', %s)"
    cursor.execute(query, (text,))
    return cursor.fetchone()[0]

def to_vector_literal(vec):
    return "ARRAY_CONSTRUCT(" + ",".join(str(x) for x in vec) + ")::VECTOR(FLOAT, 768)"

def retrieve_context(user_input, history_text=""):
    """Retrieve claims related to query, optionally including history."""
    # Merge history with current query for context
    full_query = f"{history_text}\n{user_input}" if history_text else user_input
    q_vec = get_embedding(full_query)
    q_vec_sql = to_vector_literal(q_vec)
    
    query = f"""
    SELECT CLAIM_ID, CLAIM_DESCRIPTION
    FROM CORTEX_FRAUD.CORTEX_FRAUD_SCHEMA.CORTEX_FRAUD_TABLE
    ORDER BY VECTOR_COSINE_SIMILARITY(CORTEX_FRAUD_VECTOR, {q_vec_sql}) DESC
    LIMIT 3
    """
    cursor.execute(query)
    results = cursor.fetchall()
    return "\n".join([r[1] for r in results])

def generate_answer(context: str, history: str, user_input: str):
    """Use Cortex COMPLETE for reasoning mode (slower)."""
    prompt = f"""
    You are an expert in insurance.
    Use the conversation history and claims context to answer.

    History:
    {history}

    Context:
    {context}

    Question: {user_input}
    """
    try:
        query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('gemma-7b',%s)"
        cursor.execute(query, (prompt,))
        return cursor.fetchone()[0]
    except Exception as e:
        return f"⚠️ Error from Cortex: {e}"

# ------------------------
# SESSION STATE
# ------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None


# ------------------------

# ------------------------
# MAIN CHAT
# ------------------------
st.title("AI-Powered Claim Summarizer")

if st.session_state.current_chat is None:
    current_messages = []
else:
    current_messages = st.session_state.chats[st.session_state.current_chat]["messages"]

for role, message in current_messages:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)

# ------------------------
# USER INPUT
# ------------------------
user_input = st.chat_input("Ask a question about insurance claims...")

if user_input:
    # Start new chat if none
    if st.session_state.current_chat is None:
        chat_name = f"{user_input[:30]}..."
        st.session_state.chats.append({"name": chat_name, "messages": []})
        st.session_state.current_chat = len(st.session_state.chats) - 1

    # Append user message
    st.session_state.chats[st.session_state.current_chat]["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show assistant placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

    # Build history for context
    history_text = "\n".join([f"{r}: {m}" for r, m in current_messages if r == "user"])

    # Retrieval
    context = retrieve_context(user_input, history_text)

    answer = generate_answer(context, history_text, user_input)


    # Update UI
    placeholder.markdown(answer)
    st.session_state.chats[st.session_state.current_chat]["messages"].append(("assistant", answer))







