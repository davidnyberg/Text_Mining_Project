import streamlit as st
import gpt_2_simple as gpt2
import os
import tensorflow as tf

def generate_review(review_item, confidence):
    assert(type(review_item) == str)
    confidence = float(confidence)
    confidence = 0.0 if confidence <= 0.0 else confidence
    confidence = 1.0 if confidence > 1.0 else confidence

    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        gpt2.download_gpt2(model_name=model_name)
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='fine_tune_1')

    result = gpt2.generate(sess,
                  run_name='fine_tune_1',
                  length=200,
                  temperature= confidence,
                  prefix=review_item,
                  nsamples=1,
                  return_as_list=True
                  )
    return result[0].split('\n')[0]



st.title('Fake Review Generator')
conf = st.slider("Choose how flavorful reviews you want (0 = tame, 1 = exciting)", 0.0, 1.0, 0.5, 0.1)
user_input = st.text_input("Enter a prompt to generate a review about", "food")

if st.button('Generate Review'):
    result = generate_review(user_input, conf)
    st.write(result)

