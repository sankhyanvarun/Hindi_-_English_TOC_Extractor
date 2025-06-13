import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import os

# Configure paths - CRITICAL FIX
if os.path.exists('/usr/bin/tesseract'):  # Streamlit Cloud
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    poppler_path = '/usr/bin'
else:  # Local development
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    poppler_path = None

st.title("PDF TOC Extractor")

# Debug info
st.subheader("System Verification")
st.write(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
st.write(f"Poppler path: {poppler_path}")

try:
    st.write("Tesseract version:", pytesseract.get_tesseract_version())
    st.success("Tesseract is working!")
except:
    st.error("Tesseract not found!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        st.success(f"Converted {len(images)} pages to images")
        
        # Simple OCR test
        text = pytesseract.image_to_string(images[0])
        st.text_area("OCR Output", text, height=200)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        os.unlink(pdf_path)
