import streamlit as st

st.set_page_config(page_title="Diffusion", layout="wide")

st.markdown(
    """
   <style>
   [data-testid="stSidebar"][aria-expanded="true"]{
       min-width: 450px;
       max-width: 450px;
   }
   """,
    unsafe_allow_html=True,
)

st.title("Diffusion Playground")

st.sidebar.info("Select a demo above.")

st.markdown("Diffusion WebUI built with [Streamlit](https://streamlit.io/).")
