import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import os
import tempfile
import PyPDF2
import numpy as np

# Configure paths
poppler_path = '/usr/bin' if os.path.exists('/usr/bin') else None
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' if os.path.exists('/usr/bin/tesseract') else 'tesseract'

# Enhancement function
def enhance_image(img):
    img = img.convert('L')
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.point(lambda p: 0 if p < 150 else 255)
    return img

def extract_text_from_pdf(pdf_path, language='eng'):
    try:
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            poppler_path=poppler_path,
            dpi=300,
            grayscale=True,
            thread_count=4
        )
        
        if not images:
            st.error("No pages converted")
            return ""
            
        # Process first page only for demo
        img = images[0]
        enhanced_img = enhance_image(img)
        
        # OCR with language selection
        lang = 'hin+eng' if language == 'Hindi' else 'eng'
        text = pytesseract.image_to_string(
            enhanced_img,
            lang=lang,
            config='--psm 6 --oem 3'
        )
        
        return text
        
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return ""

# Streamlit UI
st.title("PDF TOC Extractor")

# System check
st.subheader("System Verification")
try:
    st.write(f"Tesseract: {pytesseract.get_tesseract_version()}")
    st.write(f"Poppler: {'Found' if poppler_path else 'Not found'}")
except:
    st.error("System verification failed")

# Language selection
language = st.radio("Document Language", ["English", "Hindi"])

# File upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    try:
        # Create temporary file with proper permissions
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_path, language)
        
        if text:
            st.success(f"Extracted {len(text)} characters")
            with st.expander("View Extracted Text"):
                st.text(text)
            
            # Simple TOC detection
            if ("contents" in text.lower() or 
                "table of contents" in text.lower() or 
                "सूची" in text or 
                "विषय" in text):
                st.success("Found TOC indicators in text")
            else:
                st.warning("No clear TOC indicators found")
        else:
            st.error("No text extracted")
            
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
