import streamlit as st
import snowflake.connector

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Boolean Chat_new_2", layout="wide")

st.markdown("""
    <style>
    /* Sidebar as full-height flex column */
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh; 
        display: flex;
        flex-direction: column;
    }

    /* Top (logo) */
    .sidebar-top {
        flex: 0 0 auto;   /* fixed at top */
        text-align: center;
        padding: 10px 0;
    }

    /* Middle (About Us) */
    .sidebar-middle {
        flex: 1 0 auto;   /* take remaining space */
        display: flex;
        justify-content: center;  /* center horizontally */
        align-items: center;      /* center vertically */
        text-align: center;
        padding: 50px;
    }

    /* Bottom (social icons) */
    .sidebar-bottom {
        flex: 0 0 auto;   /* fixed at bottom */
        text-align: center;
        padding: 30px 30px;
        display: flex;
        justify-content: center;
        gap: 15px;
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
# SNOWFLAKE CONNECTION
# ------------------------

import streamlit as st
import snowflake.connector

conn = snowflake.connector.connect(
    account=st.secrets["snowflake"]["account"],
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    role=st.secrets["snowflake"]["role"],
    warehouse=st.secrets["snowflake"]["warehouse"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"]
)
cursor = conn.cursor()

# ------------------------
# INITIALIZE SESSION STATE
# ------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# ------------------------
# MAIN CHAT AREA
# ------------------------
st.title("AI-Powered Claim Summarizer")

if st.session_state.current_chat is None:
    current_messages = []
else:
    current_messages = st.session_state.chats[st.session_state.current_chat]["messages"]

# Render chat messages
for role, message in current_messages:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)

# ------------------------
# FIXED BOTTOM INPUT BAR
# ------------------------
user_input = st.chat_input("Ask a question about insurance claims...")

if user_input:
    # Start new chat if needed
    if st.session_state.current_chat is None:
        chat_name = f"{user_input[:30]}..."
        st.session_state.chats.append({"name": chat_name, "messages": []})
        st.session_state.current_chat = len(st.session_state.chats) - 1

    # Append user message & render
    st.session_state.chats[st.session_state.current_chat]["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------------
    # RUN RAG QUERY IN SNOWFLAKE
    # ------------------------
    query = f"""
    WITH query AS (
      SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
        'snowflake-arctic-embed-m',
        '{user_input}'
      ) AS q_vec
    ),
    retrieved AS (
      SELECT
        CLAIM_ID,
        CLAIM_DESCRIPTION
      FROM CORTEX_FRAUD.CORTEX_FRAUD_SCHEMA.CORTEX_FRAUD_TABLE, query
      ORDER BY VECTOR_COSINE_SIMILARITY(CORTEX_FRAUD_VECTOR, q_vec) DESC
      LIMIT 5
    ),
    context AS (
      SELECT LISTAGG(CLAIM_DESCRIPTION, '\\n') AS ctx
      FROM retrieved
    )
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'claude-3-5-sonnet',
        CONCAT(
          'You are an expert in insurance. Using the following context from claim data:\\n\\n',
          ctx,
          '\\n\\nAnswer this question clearly: {user_input}'
        )
    ) AS answer
    FROM context;
    """

    result = session.sql(query).collect()
    answer = result[0]["ANSWER"]

    # ------------------------
    # Render assistant message
    # ------------------------
    st.session_state.chats[st.session_state.current_chat]["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.rerun()




