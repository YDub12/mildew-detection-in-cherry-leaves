import streamlit as st
import matplotlib.pyplot as plt

def page_summary_body():
    st.write('## Project Summary and Hypotheses')

    st.info(
        f"### Project Hypotheses\n\n"
        "The hypotheses for the project are linked to the Business Requirements."
    )

    st.write("---")

    st.info(
        "### Hypothesis 1"
        "A deep learning image classification model can distinguish between " \
        "healthy leaves and those affected by mildew."
    )

    st.success(
        "Average infected leaves show a cloudier appearance than healthy leaves" \
    )

    # Load average image for healthy and mildewed leaves
    avg_powdery_mildew = plt.imread(
        "outputs/v1/avg_var_powdery_mildew.png"
    )

    st.image(
        avg_powdery_mildew,
        caption="Infected Leaves"
    )

    avg_uninfected = plt.imread(
        "outputs/v1/avg_var_healthy.png"
    )

    st.image(
        avg_uninfected,
        caption='Healthy Leaves'
    )

    st.write("---")

    st.info(
        "### Hypothesis 2"
        "Automating the leaf diagnosis process will lead to reduced manual labour"
        "and overall time spent vs. human inspection"
    )

    st.write("Manual inspection of leaves in agricultural settings is time-consuming, " \
    "subjective, and labour-intensive. By implementing a machine learning model " \
    "that can process and classify images in seconds, we significantly reduce " \
    "the need for human intervention. The trained model enables rapid batch " \
    "evaluation of leaves using only a camera or smartphone, accelerating the " \
    "diagnosis workflow and freeing up resources for more strategic tasks in " \
    "crop management.")
    

    st.write("---")

    st.info(
        "### Hypothesis 3"
        "Model performance will generalise well across unseen data"
    )

    st.write(
        "To evaluate generalization, the model was tested on a validation set" \
        "unseen during training")