import streamlit as st
from streamlit_option_menu import option_menu

def render_top_menu(current_page_name):
    # 1. FIXED DARK THEME CSS
    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #FAFAFA; }
            [data-testid="stSidebar"] { display: none; } 
            [data-testid="stHeader"] { display: none; }
            footer { visibility: hidden; }
            .block-container { padding-top: 2rem; }
            h1, h2, h3, p, span { color: #FAFAFA !important; }
        </style>
    """, unsafe_allow_html=True)

    # 2. RENDER THE PREMIUM MENU
    pages = ["Home", "Customer Hub", "Demand Explorer", "CRM Action Center"]
    icons = ["house", "people-fill", "graph-up-arrow", "bullseye"]
    
    try:
        default_idx = pages.index(current_page_name)
    except ValueError:
        default_idx = 0

    selected = option_menu(
        menu_title=None,  
        options=pages, 
        icons=icons, 
        menu_icon="cast", 
        default_index=default_idx, 
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"}, 
            "icon": {"color": "#F59E0B", "font-size": "20px"}, 
            "nav-link": {"color": "#FAFAFA", "font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#3A3B45"},
            "nav-link-selected": {"background-color": "#0f172a"},
        }
    )

    if selected != current_page_name:
        if selected == "Home": st.switch_page("app.py")
        elif selected == "Customer Hub": st.switch_page("pages/1_customer_hub.py")
        elif selected == "Demand Explorer": st.switch_page("pages/2_demand_explorer.py")
        elif selected == "CRM Action Center": st.switch_page("pages/3_crm_action_center.py")