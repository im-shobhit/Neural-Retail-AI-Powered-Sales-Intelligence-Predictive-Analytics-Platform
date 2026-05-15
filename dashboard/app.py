import streamlit as st
import sys
import os

# 1. ADD THIS CSS FIRST (To fix the login page appearance)
st.set_page_config(page_title="NeuralRetail | Platform", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;} 
        [data-testid="stHeader"] {display: none;}  
        footer {visibility: hidden;}               
        .block-container {padding-top: 5rem;}
    </style>
""", unsafe_allow_html=True)

# 2. NOW continue with your navigation import and password logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from navigation import render_top_menu

def check_password():
    def password_entered():
        if st.session_state["password"] == "admin2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:
            st.title("NeuralRetail AI Platform")
            st.markdown("Please enter your credentials to access the analytics suite.")
            st.text_input("Password (hint: admin2026)", type="password", on_change=password_entered, key="password")
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("😕 Password incorrect. Access denied.")
        return False
    else:
        return True

if check_password():
    # --- THIS IS ALL WE NEED NOW! ---
    render_top_menu("Home") 
    
    st.title("Welcome to NeuralRetail")
    st.markdown("### Enterprise AI Sales Intelligence Platform")
    st.success("Authentication successful. Role: Executive Admin.")
    st.info("👆 Please use the premium navigation menu above to access your AI modules.")