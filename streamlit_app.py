import os
from nltk.tokenize import sent_tokenize
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import inference
import config

dirname = os.path.dirname(__file__)

geclec_t5_path = os.path.join(dirname, 'models', config.T5_GEC_LEC)
geclec_t5_tok = AutoTokenizer.from_pretrained(geclec_t5_path)
geclec_t5_model = AutoModelForSeq2SeqLM.from_pretrained(geclec_t5_path)

st.title('GEC + LEC')

query_params = st.experimental_get_query_params()
value = ""
if query_params and not value:
    value = query_params['input'][0]

text = st.text_area('Input', placeholder="The main objectives of this colleges are to help people learn knowledge, "
                                                  "to strengthen their cultural competences, "
                                                  "and to participate in the growth of nearby industry.", value=value)


def sents_inference(txt):
    sents = sent_tokenize(txt)
    if len(sents) > 1:
        return inference.correct_many_sents(model=geclec_t5_model, tok=geclec_t5_tok, sent_list=sents)
    else:
        return inference.correct_sent(model=geclec_t5_model, tok=geclec_t5_tok, sent=sents[0])


if st.button('Correct This'):
    res = sents_inference(txt=text)
    st.experimental_set_query_params(input=text)
    st.write(res)


