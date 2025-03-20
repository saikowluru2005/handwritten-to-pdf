import streamlit as st
import pytesseract
from PIL import Image
from fpdf import FPDF
import io
import cv2
import numpy as np
import os

# Configure Tesseract (Ensure Tesseract is installed and its path is set correctly)
# For Windows, update the path if necessary
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    st.error("Tesseract not found! Please install Tesseract OCR and update the path.")

# def preprocess_image(image):
#     """Convert image to grayscale and apply thresholding for better OCR results."""
#     img = np.array(image)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     return Image.fromarray(thresh)


def preprocess_image(image):
    """Enhance image for better OCR results."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better results
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    return Image.fromarray(thresh)


def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    try:
        processed_image = preprocess_image(image)
        # return pytesseract.image_to_string(processed_image, lang="eng")
        return pytesseract.image_to_string(image, lang="eng+hin")  # Example: English + Hindi

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def create_pdf(text, output_filename="output.pdf"):
    """Create a PDF from extracted text."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, dest='S')
    return pdf_output.getvalue()

def main():
    st.title("Handwritten Page to Typed PDF Converter")
    st.write("Upload an image of a handwritten page to convert it into a typed PDF.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Convert to PDF"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)
                
                if extracted_text.strip():
                    pdf_bytes = create_pdf(extracted_text)
                    st.success("Conversion successful! Download your PDF below.")
                    st.download_button(label="Download PDF", data=pdf_bytes, file_name="converted.pdf", mime="application/pdf")
                else:
                    st.error("No text detected. Please upload a clearer image.")

if __name__ == "__main__":
    main()