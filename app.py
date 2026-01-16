import streamlit as st
import numpy as np
from PIL import Image
import logic

st.set_page_config(layout="wide", page_title="Test Screenshot Converter")

st.title("📸 Test Screenshot to Text Converter")
st.markdown("""
Upload a screenshot of a multiple-choice test (especially dark mode with green correct answer).
The app will extract the text and identify the correct answer.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Use session state to prevent re-processing on every keystroke in text area
    # We identify the file by name and size to be safe, or just rely on the object if Streamlit preserves it.
    # Streamlit file uploader object changes ID on rerun usually? No, it stays if not removed.

    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if "current_file_id" not in st.session_state or st.session_state.current_file_id != file_id:
        # New file detected
        file_bytes = uploaded_file.read()

        with st.spinner("Processing image... (OCR & Color Detection)"):
            try:
                result = logic.process_image(file_bytes)
                st.session_state.processing_result = result
                st.session_state.editor_text = result['full_text']
                st.session_state.current_file_id = file_id
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    # Check if we have results
    if "processing_result" in st.session_state:
        result = st.session_state.processing_result

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_container_width=True)

            st.subheader("Processed (Debug)")
            st.image(result['processed_image'], caption="Green detection & Text boxes", use_container_width=True)

        with col2:
            st.subheader("Extracted Text")

            # Editable text area
            # Key 'editor_text' binds this to st.session_state.editor_text
            text_input = st.text_area(
                "Edit text before copying:",
                key="editor_text",
                height=400
            )

            st.markdown("### Copy Result")
            st.code(st.session_state.editor_text, language="text")
            st.caption("Click the copy button in the top right of the code block above.")
