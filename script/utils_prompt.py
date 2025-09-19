
def construct_prompt_from_template(template_path, **kwargs):
    """
    Construct a prompt using a string template from a .txt or .md file.

    Args:
        template_path (str): Path to the template file (.txt or .md).
        **kwargs: Key-value pairs to populate the template.

    Returns:
        str: The constructed prompt.
    """
    # Read the template from the file
    with open(template_path, 'r') as file:
        template = file.read()

    # Format the template with the provided arguments
    prompt = template.format(**kwargs)
    return prompt
