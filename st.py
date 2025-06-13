import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import os
import tempfile
import PyPDF2
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict

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

def detect_tables(img):
    """Detect tables in an image using OpenCV"""
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(img)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Detect horizontal and vertical lines
    horizontal = cv2.erode(thresh, np.ones((1, 50), np.uint8), iterations=1)
    vertical = cv2.erode(thresh, np.ones((50, 1), np.uint8), iterations=1)
    
    # Combine lines
    mask = horizontal + vertical
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and return table bounding boxes
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:  # Filter small areas
            tables.append((x, y, x+w, y+h))
    return tables

def extract_text_from_pdf(pdf_path, language='eng', extract_tables=False):
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
            return "", []
            
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
        
        # Table extraction
        tables = []
        if extract_tables:
            # Detect tables
            table_boxes = detect_tables(enhanced_img)
            
            # Extract content from each table
            for i, (x1, y1, x2, y2) in enumerate(table_boxes):
                table_img = enhanced_img.crop((x1, y1, x2, y2))
                table_text = pytesseract.image_to_string(
                    table_img,
                    lang=lang,
                    config='--psm 6 --oem 3'
                )
                tables.append({
                    'bbox': (x1, y1, x2, y2),
                    'text': table_text.strip(),
                    'image': table_img
                })
        
        return text, tables
        
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return "", []

# Streamlit UI
st.title("PDF TOC and Table Extractor")

# System check
st.subheader("System Verification")
try:
    st.write(f"Tesseract: {pytesseract.get_tesseract_version()}")
    st.write(f"Poppler: {'Found' if poppler_path else 'Not found'}")
except:
    st.error("System verification failed")

# Language selection
language = st.radio("Document Language", ["English", "Hindi"])

# Extraction options
extract_tables = st.checkbox("Extract Tables")
extract_toc = st.checkbox("Detect TOC", value=True)

# File upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    try:
        # Create temporary file with proper permissions
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        text, tables = extract_text_from_pdf(tmp_path, language, extract_tables)
        
        if text:
            st.success(f"Extracted {len(text)} characters")
            
            # TOC detection
            toc_found = False
            if extract_toc:
                toc_keywords = ["contents", "table of contents", "सूची", "विषय"]
                if any(keyword in text.lower() for keyword in toc_keywords):
                    st.success("Found TOC indicators in text")
                    toc_found = True
                else:
                    st.warning("No clear TOC indicators found")
            
            # Display extracted text
            with st.expander("View Extracted Text"):
                st.text(text)
            
            # Display tables
            if tables:
                st.subheader(f"Detected Tables: {len(tables)}")
                for i, table in enumerate(tables):
                    with st.expander(f"Table {i+1} (Bounding Box: {table['bbox']})"):
                        st.image(table['image'], caption=f"Table {i+1}")
                        st.text(table['text'])
                        
        else:
            st.error("No text extracted")
            
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
