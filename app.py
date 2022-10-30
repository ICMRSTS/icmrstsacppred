import streamlit as st
import pandas as pd
# import shap
from keras.models import load_model
from keras.models import load_model
import gensim
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pickle
import streamlit as st






st.markdown("<h1 style='text-align: center; color: #7FFF0D;'>ICMR Combined-ACPpred</h1>", unsafe_allow_html=True)


st.write("""<h1 style='text-align: center; color: #8FBC8F;'>

**This web-app classifies the Protein Sequence into ACP/Non-ACP**
""", unsafe_allow_html=True)
st.write("""The proposed model combines the predictions of 3 different deep learning algorithms that
have proven to perform better in the supervised classification of various sequences-based
tasks. These models include ProtCNN, Attention mechanism + BiLSTM, BERT. Later every individual classifiers prediction probability is summed up based on the classifiers
weighted importance. Then finally the class having the highest weighted sum probabilities
gets the vote.""")


def user_input_features():
   
    sequence = st.text_input("Enter the Peptides Sequence", "ACCGT")
 
    data = {
        "Sequence":sequence
    }
    features = pd.DataFrame(data, index=[0])
    return features

dataframe = user_input_features()


# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(dataframe)
st.write('---')



codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
      
      char_dict = {}
      for index, val in enumerate(codes):
        char_dict[val] = index+1

      return char_dict

def integer_encoding(data, char_dict):
      
  
  encode_list = []
  for row in data:
    row_encode = []
    for code in row:
      row_encode.append(char_dict.get(code, 0))
    encode_list.append(np.array(row_encode))
  
  return encode_list



char_dict = create_dict(codes)




def transform(features):
    encode = integer_encoding(features, char_dict) 
    max_length = 100
    pad = tf.keras.preprocessing.sequence.pad_sequences(encode, maxlen=max_length, padding='post', truncating='post')
    return pad


# Build Regression Model
model = pickle.load(open('model2.pkl', 'rb'))

# Apply Model to Make Prediction
data = transform(dataframe)


prediction = model.predict(data)
prediction_proba = model.predict_proba(data)


st.subheader('Prediction')
protein_species = np.array(["ACP", "Non-ACP"])
st.write(protein_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
