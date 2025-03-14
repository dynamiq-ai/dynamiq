import os

from dynamiq import ROOT_PATH


def default_prompt_template() -> str:
    """
    Returns the default prompt template for the language model.
    """
    return r"""
            Please answer the following question based on the information found
            within the sections enclosed by triplet quotes (\`\`\`).
            Your response should be concise, well-written, and follow markdown formatting guidelines:

            - Use bullet points for list items.
            - Use **bold** text for emphasis where necessary.

            **Question:** {{query}}

            Thank you for your detailed attention to the request
            **Context information**:
            ```
            {% for document in documents %}
                ---
                Document title: {{ document.metadata["title"] }}
                Document information: {{ document.content }}
                ---
            {% endfor %}
            ```

            **User Question:** {{question}}
            Answer:
            """


def list_file_paths(folder_path=os.path.join(os.path.dirname(ROOT_PATH), "examples/data/")) -> list[str]:
    file_names = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]

    return file_paths
