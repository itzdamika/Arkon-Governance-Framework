from docx import Document
from copy import deepcopy

def reverse_docx_content(input_file, output_file):
    doc = Document(input_file)

    # Get all body elements (paragraphs, tables, etc.)
    body_elements = list(doc.element.body)

    # Reverse them
    reversed_elements = list(reversed(body_elements))

    # Create new document
    new_doc = Document()

    # Remove default empty paragraph
    new_doc.element.body.clear()

    # Add reversed content
    for element in reversed_elements:
        new_doc.element.body.append(deepcopy(element))

    new_doc.save(output_file)


# Usage
reverse_docx_content("Damika_Weekly_Log_2237951.docx", "Weekly_Log_Damika_2237951.docx")