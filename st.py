import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import re
import os
import pandas as pd
import tempfile
import PyPDF2
import numpy as np
import cv2
from typing import List, Dict
from io import BytesIO

# Set page config
st.set_page_config(page_title="Enhanced Multi-Language TOC Extractor", layout="wide")

# Initialize session state
if 'toc_df' not in st.session_state:
    st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'language' not in st.session_state:
    st.session_state.language = "English"  # Default to English for this format
if 'extra_pages' not in st.session_state:
    st.session_state.extra_pages = 2
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'new_col_name' not in st.session_state:
    st.session_state.new_col_name = ""
if 'new_col_default' not in st.session_state:
    st.session_state.new_col_default = ""
if 'max_pages' not in st.session_state:
    st.session_state.max_pages = 70
if 'start_page' not in st.session_state:
    st.session_state.start_page = 1
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# Configure paths for cloud compatibility
poppler_path = '/usr/bin' if os.path.exists('/usr/bin') else None
tesseract_path = '/usr/bin/tesseract' if os.path.exists('/usr/bin/tesseract') else 'tesseract'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Hindi digit map
HINDI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]

# ========== COMMON FUNCTIONS ==========
def get_total_pages(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return 0

def enhance_image(img):
    """Apply advanced image enhancement for better OCR results"""
    try:
        # Convert to grayscale
        img = img.convert('L')
        
        # Convert to OpenCV format for advanced processing
        cv_img = np.array(img)
        
        # Apply denoising - use a stronger denoising for old documents
        cv_img = cv2.fastNlMeansDenoising(cv_img, None, h=20, templateWindowSize=7, searchWindowSize=21)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cv_img = clahe.apply(cv_img)
        
        # Apply sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        cv_img = cv2.filter2D(cv_img, -1, kernel)
        
        # Apply dilation to thicken text
        kernel = np.ones((1, 1), np.uint8)
        cv_img = cv2.dilate(cv_img, kernel, iterations=1)
        
        # Convert back to PIL image
        img = Image.fromarray(cv_img)
        
        # Apply PIL-based enhancements
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
    except Exception:
        # Fallback to basic enhancement if OpenCV fails
        img = img.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.5)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
    
    # Apply adaptive thresholding
    img = img.point(lambda p: 0 if p < 180 else 255)
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

def ocr_from_image(image: Image, language: str) -> str:
    """Perform OCR on an image"""
    try:
        enhanced_img = enhance_image(image)
        # Use higher quality OCR configuration
        config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        lang = "hin+eng" if language == "Hindi" else "eng"
        text = pytesseract.image_to_string(enhanced_img, config=config, lang=lang)
        if language == "Hindi":
            return normalize_hindi_digits(text)
        return text
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

# ========== TEXT EXTRACTION FUNCTIONS ==========
def extract_page_text(pdf_path: str, page_num: int, language: str) -> str:
    """Extract text from a single PDF page with enhanced OCR"""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=poppler_path,
            dpi=350,  # Higher DPI for better quality
            grayscale=True,
            thread_count=4
        )
        if images:
            img = images[0]
            return ocr_from_image(img, language) + "\n"
    except Exception as e:
        st.error(f"PDF Processing Error on page {page_num+1}: {str(e)}")
        return ""
    return ""

def extract_text_from_pages(pdf_path: str, page_indices: List[int], language: str) -> str:
    """Extract text from multiple PDF pages with OCR"""
    accumulated = ""
    for idx in page_indices:
        page_text = extract_page_text(pdf_path, idx, language)
        if page_text:
            accumulated += page_text
    return accumulated

# ========== IMPROVED TOC EXTRACTION ==========
def is_valid_roman_numeral(s: str) -> bool:
    """Check if string is a valid Roman numeral (any length)"""
    # Basic pattern for Roman numerals (I, V, X, L, C, D, M)
    roman_pattern = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_pattern, s.upper()))

def is_valid_page_number(candidate: str) -> bool:
    """Check if candidate is a valid page number (digit or Roman numeral)"""
    # Check for digits
    if candidate.isdigit():
        return True
    
    # Check for Roman numerals
    if is_valid_roman_numeral(candidate):
        return True
    
    # Check for Roman numerals with common OCR errors
    if candidate.upper() in {'V', 'X', 'L', 'C', 'D', 'M'}:
        return True
    
    return False

def parse_toc_english(text: str) -> List[Dict[str, str]]:
    """Parse TOC from English text with enhanced handling for complex formats"""
    entries: List[Dict[str, str]] = []
    skip_terms = ["table of contents", "contents", "page", "toc", "chapter", "page no"]
    current_entry_lines = []  # Collect lines for the current TOC entry
    min_title_length = 4  # Minimum characters to consider as valid title
    current_section = None  # Track current section (like CHAPTER I)

    # Enhanced pattern to match page numbers (including Roman numerals)
    # Requires separator before page number and minimum distance from start
    page_num_pattern = r'[\s\.\-\:;]+([ivxlcdmIVXLCDM]+|\d+)[\s.]*$'

    # Pattern to match section headers like "CHAPTER I"
    section_pattern = r'^(CHAPTER|PART|SECTION)\s*[IVXLCDM0-9]+$'

    # Normalize text: replace common OCR errors
    replacements = {
        '¥': 'v',  # Common OCR error for Roman numeral v
        ' . ': '.',
        '..': '.',
        ',,': ',',
        ' :': ':',
        ' ;': ';',
        ' -': '-',
        '—': '-',
        ' .': '.',
        ' ,': ',',
        '  ': ' ',
        '\'': '',  # Remove apostrophes that might cause word breaks
        '‘': '',   # Remove smart quotes
        '’': '',
        '“': '',
        '”': '',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        
        # Skip if the line contains no word characters
        if not re.search(r'\w', line):
            continue
        
        # Check if this is a section header
        if re.match(section_pattern, line, re.IGNORECASE):
            current_section = line
            continue
        
        # Check if this line should be skipped because it's a header
        lower_line = line.lower()
        if any(term in lower_line for term in skip_terms):
            continue
        
        # Check if this line ends with a sequence of digits or Roman numerals (page number)
        page_match = re.search(page_num_pattern, line)
        if page_match:
            page_str = page_match.group(1).strip()
            
            # Validate page number - only allow digits or Roman numerals
            if not is_valid_page_number(page_str):
                # Not a valid page number - treat as regular line
                current_entry_lines.append(line)
                continue
            
            # The title part is the string without the page number and separator
            title_part = line[:page_match.start()].strip()
            
            # Remove trailing dots, dashes, colons, etc.
            title_part = re.sub(r'[\.\-\:\;\s]+$', '', title_part)
            
            # Prepend current section if exists
            full_title = ""
            if current_section:
                full_title = current_section + " - "
                current_section = None
                
            # Prepend any accumulated lines
            if current_entry_lines:
                full_title += " ".join(current_entry_lines) + " "
                current_entry_lines = []
                
            full_title += title_part
            
            # Clean up excessive spaces
            full_title = re.sub(r'\s+', ' ', full_title).strip()
            
            # Skip if the full_title is too short or is a skip term
            if len(full_title) < min_title_length or any(term in full_title.lower() for term in skip_terms):
                continue
                
            # Format Roman numerals consistently
            if page_str.isdigit():
                page_num = page_str
            else:
                page_num = page_str.upper()  # Standardize to uppercase Roman numerals
                
            entries.append({"Title": full_title, "Page": page_num})
        else:
            # Line doesn't end with page number → buffer it
            # But check if it might be a section header
            if re.match(r'^[A-Z\s\-]+$', line):
                # Likely a section header or chapter title
                current_section = line
            else:
                current_entry_lines.append(line)
            
    # After processing all lines, check if there are any remaining lines in the buffer
    if current_entry_lines:
        full_title = " ".join(current_entry_lines).strip()
        if len(full_title) >= min_title_length and not any(term in full_title.lower() for term in skip_terms):
            entries.append({"Title": full_title, "Page": "?"})
            
    return entries

# ========== UI AND MAIN APP ==========
def main():
    st.title("📖 PDF & Image TOC Extractor (Hindi + English)")
    
    # System verification
    st.subheader("System Verification")
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        st.success(f"Tesseract OCR {tesseract_version} is ready!")
    except:
        st.error("Tesseract not properly configured!")
    
    # Language selection
    st.session_state.language = st.radio(
        "Select Document Language:",
        ["Hindi", "English"],
        horizontal=True,
        index=1  # Default to English
    )
    
    # Configuration
    with st.expander("Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.extra_pages = st.slider(
                "Extra pages after TOC (PDF only)",
                min_value=0,
                max_value=20,
                value=st.session_state.extra_pages,
                help="TOC might span multiple pages"
            )
        with col2:
            st.session_state.max_pages = st.slider(
                "Max pages for large PDFs",
                min_value=5,
                max_value=200,
                value=st.session_state.max_pages,
                help="For large PDFs, only process first N pages"
            )
        
        # Starting page selection (PDF only)
        st.session_state.start_page = st.number_input(
            "Starting page for extraction (PDF only, 1-based)",
            min_value=1,
            max_value=1000,
            value=st.session_state.start_page,
            help="The page number from which to start searching for TOC"
        )
    
    # File upload section - now supports images too
    uploaded_file = st.file_uploader(
        "Upload PDF or Image file", 
        type=["pdf"] + SUPPORTED_IMAGE_FORMATS
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_image = file_extension in SUPPORTED_IMAGE_FORMATS
        
        # Display file info
        st.info(f"Uploaded file: {uploaded_file.name} ({'Image' if is_image else 'PDF'})")
        st.session_state.file_type = "image" if is_image else "pdf"
        
        if st.button("Extract Table of Contents"):
            with st.spinner("Processing file..."):
                try:
                    toc_entries = []
                    
                    if is_image:
                        # Process image file
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        # Perform OCR
                        extracted_text = ocr_from_image(image, st.session_state.language)
                        st.session_state.raw_text = extracted_text
                        
                        # Extract TOC based on language
                        toc_entries = parse_toc_english(extracted_text) or []
                    else:
                        # Process PDF file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            pdf_path = os.path.join(temp_dir, "uploaded.pdf")
                            with open(pdf_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Check if PDF is large
                            file_size = os.path.getsize(pdf_path)
                            is_large_pdf = file_size > 4 * 1024 * 1024  # 4 MB
                            truncated_path = None
                            
                            if is_large_pdf:
                                st.info(f"Large PDF detected ({file_size/(1024*1024):.2f} MB). Processing first {st.session_state.max_pages} pages.")
                                truncated_path = os.path.join(temp_dir, "truncated.pdf")
                                truncate_pdf(pdf_path, truncated_path, max_pages=st.session_state.max_pages)
                                extraction_path = truncated_path
                            else:
                                extraction_path = pdf_path
                            
                            # Language-specific extraction
                            start_page_index = st.session_state.start_page - 1  # Convert to 0-based index
                            
                            # Find TOC pages
                            toc_indices = list(range(start_page_index, start_page_index + st.session_state.max_pages))
                            
                            # Extract text from identified pages
                            extracted_text = extract_text_from_pages(
                                extraction_path, 
                                toc_indices,
                                st.session_state.language
                            )
                            st.session_state.raw_text = extracted_text
                            
                            # Use the improved parser
                            toc_entries = parse_toc_english(extracted_text) or []
                    
                    # Process results
                    if toc_entries:
                        st.success(f"Successfully extracted {len(toc_entries)} TOC entries!")
                        st.session_state.toc_df = pd.DataFrame(toc_entries)
                    else:
                        st.warning("No TOC found in the document")
                        st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
                        
                    # Show raw text extraction message
                    if st.session_state.file_type == "pdf":
                        st.info(f"Extracted {len(st.session_state.raw_text)} characters from PDF")
                    else:
                        st.info(f"Extracted {len(st.session_state.raw_text)} characters from image")
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
    
    # Edit TOC Section
    if not st.session_state.toc_df.empty:
        st.subheader("Table of Contents")
        
        # Display non-editable preview
        st.dataframe(st.session_state.toc_df, use_container_width=True)
        
        # Edit mode toggle
        if not st.session_state.edit_mode:
            if st.button("✏️ Edit TOC"):
                st.session_state.edit_mode = True
                st.session_state.backup_df = st.session_state.toc_df.copy()
        else:
            st.subheader("Edit Mode")
            
            # Display editable dataframe
            edited_df = st.data_editor(
                st.session_state.toc_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Title": st.column_config.TextColumn("Chapter Title", width="large"),
                    "Page": st.column_config.TextColumn("Page Number", width="small")
                }
            )
            
            # Edit controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Changes"):
                    st.session_state.toc_df = edited_df
                    st.session_state.edit_mode = False
                    st.success("Changes saved!")
            with col2:
                if st.button("❌ Discard Changes"):
                    st.session_state.toc_df = st.session_state.backup_df
                    st.session_state.edit_mode = False
                    st.info("Changes discarded")
            
            # Advanced editing options
            with st.expander("🔄 Advanced Editing Tools", expanded=True):
                # Row insertion at specific position
                st.subheader("Insert Row at Specific Position")
                with st.form("row_insert_form"):
                    insert_position = st.number_input(
                        "Row Position (1-based)",
                        min_value=1,
                        max_value=len(edited_df)+1,
                        value=len(edited_df)+1,
                        help="Position where the new row will be inserted"
                    )
                    new_title = st.text_input("Title")
                    new_page = st.text_input("Page")
                    
                    if st.form_submit_button("Insert Row"):
                        # Create new row as DataFrame
                        new_row = pd.DataFrame([[new_title, new_page]], columns=edited_df.columns)
                        
                        # Convert 1-based position to 0-based index
                        pos_index = insert_position - 1
                        
                        # Insert row
                        top_part = edited_df.iloc[:pos_index]
                        bottom_part = edited_df.iloc[pos_index:]
                        edited_df = pd.concat([top_part, new_row, bottom_part]).reset_index(drop=True)
                        st.session_state.toc_df = edited_df
                        st.success(f"Inserted new row at position {insert_position}")
                
                # Column insertion
                st.subheader("Add New Column")
                with st.form("column_insert_form"):
                    col_name = st.text_input("Column Name", value=st.session_state.new_col_name)
                    default_value = st.text_input("Default Value", value=st.session_state.new_col_default)
                    
                    if st.form_submit_button("Add Column"):
                        if col_name in edited_df.columns:
                            st.error(f"Column '{col_name}' already exists!")
                        else:
                            # Add new column with default value
                            edited_df[col_name] = default_value
                            st.session_state.toc_df = edited_df
                            st.session_state.new_col_name = col_name
                            st.session_state.new_col_default = default_value
                            st.success(f"Added new column: {col_name}")
    
    # Handle case where DataFrame is empty
    elif not st.session_state.toc_df.empty and st.session_state.toc_df.columns.tolist() != ["Title", "Page"]:
        st.warning("TOC data is in an invalid format. Resetting...")
        st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
    
    # Raw text section
    if 'raw_text' in st.session_state and st.session_state.raw_text:
        with st.expander("View Raw Extracted Text"):
            st.text_area("Raw OCR Output", st.session_state.raw_text, height=300)
            st.info(f"Total characters: {len(st.session_state.raw_text)}")
    
    # Download section
    if not st.session_state.toc_df.empty:
        st.subheader("Download Final TOC")
        csv = st.session_state.toc_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download TOC as CSV",
            data=csv,
            file_name="table_of_contents.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
