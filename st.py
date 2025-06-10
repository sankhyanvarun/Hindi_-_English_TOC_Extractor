import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import re
import os
import pandas as pd
import tempfile
import PyPDF2
import numpy as np
from typing import List, Dict

# Set page config
st.set_page_config(page_title="Enhanced Multi-Language TOC Extractor", layout="wide")

# Initialize paths based on environment
def setup_paths():
    if os.path.exists('/usr/bin/tesseract'):  # Streamlit Cloud environment
        return {
            'poppler_path': '/usr/bin',
            'tesseract_path': '/usr/bin/tesseract'
        }
    else:  # Local development
        return {
            'poppler_path': 'poppler/bin',
            'tesseract_path': 'Tesseract-OCR/tesseract.exe'
        }

paths = setup_paths()

# Initialize session state
if 'toc_df' not in st.session_state:
    st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'language' not in st.session_state:
    st.session_state.language = "Hindi"
if 'poppler_path' not in st.session_state:
    st.session_state.poppler_path = paths['poppler_path']
if 'tesseract_path' not in st.session_state:
    st.session_state.tesseract_path = paths['tesseract_path']
if 'extra_pages' not in st.session_state:
    st.session_state.extra_pages = 2
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'new_col_name' not in st.session_state:
    st.session_state.new_col_name = ""
if 'new_col_default' not in st.session_state:
    st.session_state.new_col_default = ""

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = st.session_state.tesseract_path

# Hindi digit map
HINDI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# ========== COMMON FUNCTIONS ==========
def get_total_pages(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return 0

def enhance_image(img):
    """Apply image enhancement for better OCR results"""
    img = img.convert('L')  # Convert to grayscale
    img = img.filter(ImageFilter.MedianFilter())  # Reduce noise
    enhancer = ImageEnhance.Contrast(img)  # Enhance contrast
    img = enhancer.enhance(2)  # Increase contrast
    img = img.point(lambda p: 0 if p < 150 else 255)  # Thresholding
    return img

def truncate_pdf(input_path: str, output_path: str, max_pages: int = 70) -> None:
    reader = PyPDF2.PdfReader(input_path)
    writer = PyPDF2.PdfWriter()
    
    num_pages = len(reader.pages)
    pages_to_keep = min(max_pages, num_pages)
    
    for i in range(pages_to_keep):
        writer.add_page(reader.pages[i])
    
    with open(output_path, "wb") as f:
        writer.write(f)

def normalize_hindi_digits(text):
    return ''.join(HINDI_DIGIT_MAP.get(ch, ch) for ch in text)

def check_tesseract():
    try:
        langs = pytesseract.get_languages(config='')
        st.info(f"Available Tesseract languages: {langs}")
        return 'hin' in langs and 'eng' in langs
    except Exception as e:
        st.error(f"Tesseract check failed: {str(e)}")
        return False

# ========== TEXT EXTRACTION FUNCTIONS ==========
def extract_page_text(pdf_path: str, page_num: int, language: str) -> str:
    """Extract text from a single page with OCR and enhancement"""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=st.session_state.poppler_path,
            dpi=300 if language == "Hindi" else 400,
            grayscale=True,
        )
        if images:
            img = images[0]
            enhanced_img = enhance_image(img)
            config = r'--oem 1 --psm 6'
            lang = "hin+eng" if language == "Hindi" else "eng"  # Combined for better accuracy
            page_text = pytesseract.image_to_string(enhanced_img, config=config, lang=lang)
            
            if language == "Hindi":
                return normalize_hindi_digits(page_text) + "\n"
            return page_text + "\n"
    except Exception as e:
        st.error(f"OCR Error on page {page_num}: {str(e)}")
        return ""
    return ""

def extract_text_from_pages(pdf_path: str, page_indices: List[int], language: str) -> str:
    """Extract text from multiple pages with OCR"""
    accumulated = ""
    for idx in page_indices:
        accumulated += extract_page_text(pdf_path, idx, language)
    return accumulated

# ========== HINDI-SPECIFIC FUNCTIONS ==========
def find_toc_page_indices_hindi(pdf_path: str, max_search_pages: int = 20) -> List[int]:
    """Find pages with Hindi TOC keywords"""
    indices: List[int] = []
    try:
        # Hindi TOC keywords
        toc_keywords = ["विषय सूची", "अनुक्रमणिका", "सूची", "विषय-सूची", "अनुक्रम", "सामग्री"]

        for i in range(max_search_pages):
            page_text = extract_page_text(pdf_path, i, "Hindi")
            if any(keyword in page_text for keyword in toc_keywords):
                indices.append(i)
                
        return indices

    except Exception as e:
        st.error(f"Error finding Hindi TOC pages: {e}")
        return []

def extract_toc_hindi(text):
    """Extract TOC from Hindi text"""
    toc_start_pattern = r"(विषय[-\s]*सूची|अनुक्रमणिका|सूची|पृष्ठ|अध्याय|अंक|प्रकरण|सामग्री)"
    match = re.search(toc_start_pattern, text, re.IGNORECASE)
    if not match:
        return None

    toc_section = text[match.start():]
    toc_section = re.sub(r'[^\S\r\n]+', ' ', toc_section)  # Normalize whitespace
    lines = toc_section.split('\n')

    toc_entries = []
    current_title_lines = []
    digit_pattern = r'[\d०१२३४५६७८९]+$'  # Combined digit pattern

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        # Match page number at end of line
        page_match = re.search(digit_pattern, clean_line)
        page_number = page_match.group() if page_match else None

        if page_number:
            title_part = clean_line.rsplit(page_number, 1)[0].strip("–-—:. ")
            
            if current_title_lines and len(title_part) < 5:
                title = ' '.join(current_title_lines).strip()
                toc_entries.append({
                    "Title": title,
                    "Page": normalize_hindi_digits(page_number)
                })
                current_title_lines = []
            else:
                if current_title_lines:
                    title = ' '.join(current_title_lines).strip()
                    toc_entries.append({
                        "Title": title,
                        "Page": normalize_hindi_digits(page_number)
                    })
                    current_title_lines = []
                if title_part:
                    toc_entries.append({
                        "Title": title_part,
                        "Page": normalize_hindi_digits(page_number)
                    })
        else:
            current_title_lines.append(clean_line)

    if current_title_lines:
        title = ' '.join(current_title_lines).strip()
        if len(title) > 5:
            toc_entries.append({"Title": title, "Page": "?"})

    filtered_entries = [
        entry for entry in toc_entries
        if len(entry["Title"]) >= 5 and not re.match(r'^\d+$', entry["Title"])
    ]
    
    return filtered_entries

# ========== ENGLISH-SPECIFIC FUNCTIONS ==========
def find_toc_page_indices_english(pdf_path: str, max_search_pages: int = 20) -> List[int]:
    indices: List[int] = []
    try:
        for i in range(max_search_pages):
            page_text = extract_page_text(pdf_path, i, "English")
            lower_all = page_text.lower()
            
            toc_keywords = ["contents", "table of contents", "toc", "index", "chapters"]
            if any(keyword in lower_all for keyword in toc_keywords):
                possible_entries = parse_toc_english(page_text)
                if len(possible_entries) >= 2:
                    indices.append(i)

        return indices

    except Exception as e:
        st.error(f"Error finding TOC pages: {e}")
        return []

def parse_toc_english(text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    skip_terms = ["table of contents", "contents", "page", "toc", "chapter"]
    current_entry_lines = []

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        
        cleaned = normalize_hindi_digits(line)
        
        if re.search(r'\d+\s*$', cleaned):
            full_text = ""
            if current_entry_lines:
                full_text = " ".join(current_entry_lines) + " " + cleaned
                current_entry_lines = []
            else:
                full_text = cleaned
                
            lower_text = full_text.lower()
            if any(term in lower_text for term in skip_terms):
                continue
                
            m = re.match(r'^(.*?)[\s\.\-]+\s*(\d+)\s*$', full_text)
            if not m:
                m = re.match(r'^(.*?)(\d+)\s*$', full_text)
                
            if m:
                chapter = m.group(1).strip()
                page_no = m.group(2).strip()
                entries.append({"Title": chapter, "Page": page_no})
        else:
            current_entry_lines.append(cleaned)
            
    return entries

# ========== UI AND MAIN APP ==========
def main():
    st.title("📖 Enhanced Multi-Language PDF TOC Extractor")
    st.markdown("""
    Extract Table of Contents from Hindi or English PDFs:
    - **Hindi**: Uses OCR with image enhancement
    - **English**: Uses OCR with image enhancement and text extraction
    """)
    
    # Environment check
    if not check_tesseract():
        st.error("Tesseract OCR is not properly configured. Please check deployment settings.")
        return
    
    # Language selection
    st.session_state.language = st.radio(
        "Select PDF Language:",
        ["Hindi", "English"],
        horizontal=True
    )
    
    # Configuration
    with st.expander("Configuration Settings", expanded=False):
        st.info(f"Running on: {'Streamlit Cloud' if os.path.exists('/usr/bin/tesseract') else 'Local'}")
        st.write(f"Poppler path: {st.session_state.poppler_path}")
        st.write(f"Tesseract path: {st.session_state.tesseract_path}")
    
    # Extra pages setting
    st.session_state.extra_pages = st.slider(
        "Include extra pages after TOC pages",
        min_value=0,
        max_value=20,
        value=st.session_state.extra_pages,
    )
    
    # File upload section
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "uploaded.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Extract Table of Contents"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Check if PDF is large
                        file_size = os.path.getsize(pdf_path)
                        is_large_pdf = file_size > 100 * 1024 * 1024
                        truncated_path = None
                        
                        if is_large_pdf:
                            st.info(f"Large PDF detected ({file_size/(1024*1024):.2f} MB). Using first 70 pages.")
                            truncated_path = os.path.join(temp_dir, "truncated.pdf")
                            truncate_pdf(pdf_path, truncated_path, max_pages=70)
                            extraction_path = truncated_path
                        else:
                            extraction_path = pdf_path
                        
                        # Language-specific extraction
                        if st.session_state.language == "Hindi":
                            toc_indices = find_toc_page_indices_hindi(extraction_path)
                            expanded_indices = set()
                            for i in toc_indices:
                                for offset in range(0, st.session_state.extra_pages + 1):
                                    expanded_indices.add(i + offset)
                            
                            if not expanded_indices:
                                total_pages = get_total_pages(extraction_path)
                                expanded_indices = set(range(0, min(20, total_pages)))
                            
                            extracted_text = extract_text_from_pages(
                                extraction_path, 
                                sorted(expanded_indices),
                                "Hindi"
                            )
                            st.session_state.raw_text = extracted_text
                            toc_entries = extract_toc_hindi(extracted_text) or []
                        else:
                            toc_indices = find_toc_page_indices_english(extraction_path)
                            expanded_indices = set()
                            for i in toc_indices:
                                for offset in range(0, st.session_state.extra_pages + 1):
                                    expanded_indices.add(i + offset)
                            
                            if not expanded_indices:
                                total_pages = get_total_pages(extraction_path)
                                expanded_indices = set(range(0, min(20, total_pages)))
                            
                            extracted_text = extract_text_from_pages(
                                extraction_path, 
                                sorted(expanded_indices),
                                "English"
                            )
                            st.session_state.raw_text = extracted_text
                            toc_entries = parse_toc_english(extracted_text) or []
                        
                        if toc_entries:
                            st.success(f"Successfully extracted {len(toc_entries)} TOC entries!")
                            st.session_state.toc_df = pd.DataFrame(toc_entries)
                        else:
                            st.warning("No TOC found in the document")
                            st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
                            
                        st.info(f"Extracted {len(st.session_state.raw_text)} characters from {len(expanded_indices)} pages")
                            
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
    
    # Display results
    if not st.session_state.toc_df.empty:
        st.subheader("Table of Contents")
        st.dataframe(st.session_state.toc_df, use_container_width=True)
        
        # Download section
        st.subheader("Download Final TOC")
        csv = st.session_state.toc_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download TOC as CSV",
            data=csv,
            file_name="table_of_contents.csv",
            mime="text/csv"
        )
    
    # Raw text section
    if 'raw_text' in st.session_state and st.session_state.raw_text:
        with st.expander("View Raw Extracted Text"):
            st.text_area("Raw OCR Output", st.session_state.raw_text, height=300)

if __name__ == "__main__":
    main()
