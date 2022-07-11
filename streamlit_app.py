import os
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import inference
import config

dirname = os.path.dirname(__file__)

geclec_t5_path = os.path.join(dirname, 'models', config.T5_GEC_LEC)
geclec_t5_tok = AutoTokenizer.from_pretrained(geclec_t5_path)
geclec_t5_model = AutoModelForSeq2SeqLM.from_pretrained(geclec_t5_path)

st.title('GEC + LEC')

sent = st.text_area('Input Sentence', placeholder="The main objectives of this colleges are to help people learn knowledge, "
                                     "to strengthen their cultural competences, "
                                     "and to participate in the growth of nearby industry.")

if st.button('Correct Sentence'):
    res = inference.correct_sent(model=geclec_t5_model, tok=geclec_t5_tok, sent=sent)
    st.write(res)