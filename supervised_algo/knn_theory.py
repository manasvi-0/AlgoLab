import streamlit as st
import os
from PIL import Image
def render():
    st.title("K-Nearest Negihbour(KNN)-Theory")
    st.write(
        '''K - Nearest Neighbors(KNN) is a supervised machine learning algorithm generally used for classification but can also be used for regression tasks.It works by finding the "k" closest data points (neighbors) to a given input and makesa predictions based on the majority class(for classification) or the average value ( for regression).Since KNN makes no assumptions about the underlying data distribution it makes it a non-parametric and instance-based learning method.''')
    st.write('''K-Nearest Neighbors is also called as a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification it performs an action on the dataset.
        ''')
    st.header('Working of KNN')
    st.subheader('Step 1')
    st.markdown('**Initial Data**: A new point (🟨) needs to be classified among existing Class A (❇️) and Class B (🔼).')
    #image 1
    current_dir=os.path.dirname(__file__)
    image1_path=os.path.join(current_dir,'..','assets','Initial-Data.png')
    image1=Image.open(image1_path)
    st.image(image1, caption='Step 1', width=600)
    # image 2
    st.subheader('Step 2')
    st.markdown('**Calculate Distances**: Distances from the unknown point to all other points are calculated.')

    image2_path = os.path.join(current_dir, '..', 'assets', 'Step2.png')
    image2 = Image.open(image2_path)
    st.image(image2, caption='Step 2', width=600)
    # image 3
    st.subheader('Step 3')
    st.markdown('**Voting**: Based on k=3, the nearest neighbors vote, and the majority class is assigned to the new point.')
    image3_path = os.path.join(current_dir, '..', 'assets','Step3.png')
    image3= Image.open(image3_path)
    st.image(image3, caption='Step 3', width=600)
    '''putting images side-by-side
    col1,col2,col3=st.columns(3)
    with col1:
        
    with col2:
       
    with col3:
        '''
    st.header('What is K in KNN')
    st.write('In the k-Nearest Neighbours algorithm k is just a number that tells the algorithm how many nearby points or neighbors to look at when it makes a decision.')
    st.write('**Example**: Imagine you are deciding which fruit it is based on its shape and size. You compare it to fruits you already know.')
    st.markdown("""- If k = 3, the algorithm looks at the 3 closest fruits to the new one.""")
    st.markdown("""- If 2 of those 3 fruits are apples and 1 is a banana, the algorithm says the new fruit is an apple because most of its neighbors are apples.""")