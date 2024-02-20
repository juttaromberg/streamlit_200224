import streamlit as st
import pandas as pd



#2 columns: sidebar: radio, image:show image, video: show video
#upload file via UI (.csv) - basic visualizst 1 line, bar, scatter, -  2 altair
#sidebar1 = st.sidebar.columns(2)
#col1, col2 = sidebar1
#bild = "Cat.png"
#film = "https://www.youtube.com/watch?v=Hh7W_y7_fZg"
was = st.sidebar.radio(
    "Select one:", ["image", "video"],
    )
if was == "image":
    st.image("Cat.png")
else:
    st.video("https://www.youtube.com/watch?v=Hh7W_y7_fZg")

#task 3
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read the file as bytes:
    #bytes_data = uploaded_file.getvalue()
    #st.write("bytes_data")

    # To display the file contents - this example assumes it's a text file
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)



