def load_data(file_path):
    """
    Load insurance claim records from text file
    """

    with open(file_path, "r", encoding="utf-8") as f:
        records = f.readlines()

    return records


def convert_record_to_text(records):
    """
    Convert insurance records into natural language documents
    """

    documents = []

    for record in records:

        text = f"""
Insurance claim report:
Claim description: {record.strip()}
"""

        documents.append(text)

    return documents