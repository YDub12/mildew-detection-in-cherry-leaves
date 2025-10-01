import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)
from src.data_management import download_dataframe_as_csv


def image_data_body():

    st.write(
        "## Disease Prediction\n\n"
        "Please upload an image(s) of a cherry leaf to discover "
        "if it is infected with the powdery mildew disease or not. "
        "You can then download a report of the results."
    )

    st.success(
        "This page addresses Business Requirement 2."
    )

    st.write("---")

    st.success(
        "Upload an image of a cherry leaf.\n\n"
        "Drag and drop an online image or select one from "
        "your device.\n\n"
        "You may add multiple images at once."
    )

    st.warning(
        "You can download a set of infected and healthy leaves for live "
        "prediction [here] "
        "(https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).")

    st.write("---")

    images_buffer = st.file_uploader(
        ' ', accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Cherry leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(
                img_pil,
                caption=f"Image Size: {img_array.shape[1]}px width x "
                        f"{img_array.shape[0]}px height"
            )

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append(
                {
                    "Name": image.name,
                    'Result': pred_class
                }, ignore_index=True
            )

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(
                df_report), unsafe_allow_html=True)

    st.write("---")