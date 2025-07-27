import streamlit as st
#main page
st.set_page_config(
    page_title="KNN Visualizer",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title('K Nearest Neighbour')
st.subheader("Intution")
st.markdown(f"""
    <style>
    .stApp {{
        background-color:light gray;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='
    padding: 12px;
    border-left: 5px solid black;
    background-color: rgba(74, 144, 226, 0.1);
    color: inherit;
    font-style: italic;
'>
"You are the average of the five people you spend time with"<br><b>— Jim Rohn</b>
</div>
""", unsafe_allow_html=True)
#Sider options
st.sidebar.title("Algorithms")
algo = st.sidebar.selectbox("Choose an algorithm", ["KNN"])
if algo=="KNN":
    view= st.sidebar.radio("Choose View",["KNN Overview","KNN Playground"])
    if view=="KNN Overview":
        from supervised_algo import knn_theory
        knn_theory.render()
    elif view=="KNN Playground" :
        from supervised_algo import knn_visualization
        knn_visualization.render()