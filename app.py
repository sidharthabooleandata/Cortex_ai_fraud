import streamlit as st
import snowflake.connector
from cryptography.hazmat.primitives import serialization

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Boolean Chat_new_2", layout="wide")

# ------------------------
# SIDEBAR
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
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
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
@st.cache_resource
def get_connection():
    private_key_bytes = st.secrets["snowflake"]["private_key"].encode()
    private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
    return snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        role=st.secrets["snowflake"]["role"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        private_key=private_key,
    )

conn = get_connection()
cursor = conn.cursor()

# ------------------------
# HELPERS
# ------------------------
def get_embedding(text: str):
    query = "SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', %s)"
    cursor.execute(query, (text,))
    return cursor.fetchone()[0]

def to_vector_construct(vec):
    """Convert a Python list of floats into Snowflake VECTOR_CONSTRUCT_FLOAT(...) SQL"""
    return "VECTOR_CONSTRUCT_FLOAT(" + ",".join(str(x) for x in vec) + ")"


def retrieve_context(user_input):
    q_vec = get_embedding(user_input)   # Python list of floats
    
    # Convert to valid Snowflake SQL literal
    q_vec_sql = to_vector_construct(q_vec)
    
    query = f"""
    SELECT CLAIM_ID, CLAIM_DESCRIPTION
    FROM CORTEX_FRAUD.CORTEX_FRAUD_SCHEMA.CORTEX_FRAUD_TABLE
    ORDER BY VECTOR_COSINE_SIMILARITY(CORTEX_FRAUD_VECTOR, {q_vec_sql}) DESC
    LIMIT 5
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    return "\n".join([r[1] for r in results])




def generate_answer(context: str, user_input: str):
    prompt = f"""
    You are an expert in insurance.
    Use the following claims context to answer the question:

    {context}

    Question: {user_input}
    """
    query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('claude-3-haiku', %s)"
    cursor.execute(query, (prompt,))
    return cursor.fetchone()[0]

# ------------------------
# SESSION STATE
# ------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

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
    if st.session_state.current_chat is None:
        chat_name = f"{user_input[:30]}..."
        st.session_state.chats.append({"name": chat_name, "messages": []})
        st.session_state.current_chat = len(st.session_state.chats) - 1

    st.session_state.chats[st.session_state.current_chat]["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Show placeholder while waiting
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

    # RAG pipeline
    context = retrieve_context(user_input)
    answer = generate_answer(context, user_input)

    # Update placeholder with final answer
    placeholder.markdown(answer)

    st.session_state.chats[st.session_state.current_chat]["messages"].append(("assistant", answer))
    st.rerun()




