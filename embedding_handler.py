print('cohere poc')

# Import the necessary packages
import os
import base64
import cohere
from unstructured.partition.auto import partition
from unstructured.documents.elements import Text, Table, Image


# Defining the function to convert an image to a base 64 Data URL
def image_to_base64_data_url(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_type = file_extension[1:]

    with open(image_path, "rb") as f:
        enc_img = base64.b64encode(f.read()).decode("utf-8")
        enc_img = f"data:image/{file_type};base64,{enc_img}"
    return enc_img

def get_image_embedding(processed_image):
    co = cohere.ClientV2(api_key="FIfDqpRiGUXxDq6jNUKcwPEILCpoRErujyeq3mrq")
    resp = co.embed(
        model="embed-english-v3.0",
        images=[processed_image],
        input_type="image",
        embedding_types=["float"],
    )
    return resp

def extract_content(file_path):
    # Partition the document into elements
    elements = partition(file_path)

    # Separate the elements into text, tables, and images
    texts = [element for element in elements if isinstance(element, Text)]
    tables = [element for element in elements if isinstance(element, Table)]
    images = [element for element in elements if isinstance(element, Image)]

    return texts, tables, images


import fitz  # PyMuPDF
import os

def extract_content_v1(file_path):
    # Open the PDF file
    doc = fitz.open(file_path)

    texts = []
    images = []
    tables = []  # PyMuPDF does not natively extract tables, but we can handle this separately.

    for page_num in range(24,32):
        page = doc[page_num]

        # Extract text from the page
        texts.append(page.get_text())

        # Extract images from the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"output/page_{page_num + 1}_img_{img_index + 1}.{image_ext}"

            # Save the image to a file
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            images.append(image_filename)

    return texts, tables, images



# # Print extracted content
# print("Extracted Texts:")
# for text in texts:
#     print(text)

# print("\nExtracted Images:")
# for image in images:
#     print(f"Saved image: {image}")

# Note: Table extraction is not natively supported by PyMuPDF. You can use libraries like Camelot or Tabula for table extraction.

#image_path = "C2-W4-S2-V2.jpg"
#processed_image = image_to_base64_data_url(image_path)
#embedding_response = get_image_embedding(processed_image)
#print(embedding_response)
texts, tables, images = extract_content_v1("Morgan_Stanley_2022_Form_10-K.pdf")

# Print extracted content

print("Extracted Texts:")
for text in texts:
    print(text)

print("\nExtracted Tables:")
for table in tables:
    print(table)

print("\nExtracted Images:")
for image in images:
    print(image)
