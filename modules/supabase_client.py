# modules/supabase_client.py

from supabase import create_client
import streamlit as st

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_API_KEY = st.secrets["SUPABASE_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
