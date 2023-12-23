#import modal

# definition of our container image and app for deployment on Modal
# see app.py for more details


import os

def add_to_document_db(documents_json, collection=None, db=None):
    """Adds a collection of json documents to a database."""
    from pymongo import InsertOne

    import docstore

    res = None

    db = db or os.environ["MONGODB_DATABASE"]
    collection = collection or os.environ["MONGODB_COLLECTION"]
    collection = docstore.get_collection(collection, db)

    requesting, CHUNK_SIZE = [], 250

    for document in documents_json:
        requesting.append(InsertOne(document))

        if len(requesting) >= CHUNK_SIZE:
            collection.bulk_write(requesting)
            requesting = []

    if requesting:
        res = collection.bulk_write(requesting)

    assert res is not None
    return res

def enrich_metadata(pages):
    """Add our metadata: sha256 hash and ignore flag."""
    import hashlib

    referencePageDetected = False
    for page in pages:
        m = hashlib.sha256()
        m.update(page["text"].encode("utf-8", "replace"))
        page["metadata"]["sha256"] = m.hexdigest()
        if referencePageDetected or page["metadata"].get("is_endmatter"):
            page["metadata"]["ignore"] = True # Ignore the pages that are ends (bibliography, references, etc).
            referencePageDetected = True
        else:
            page["metadata"]["ignore"] = False
    return pages


def chunk_into(list, n_chunks):
    """Splits list into n_chunks pieces, non-contiguously."""
    for ii in range(0, n_chunks):
        yield list[ii::n_chunks]


def unchunk(list_of_lists):
    """Recombines a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def display_modal_image(image):
    from IPython.display import HTML
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name

    dockerfile_commands = get_image_dockerfile_commands(image)

    fmt = HtmlFormatter(style="rrt", cssclass="_pygments_code", nobackground=False)
    css_styles = fmt.get_style_defs(".output_html")

    lexer = get_lexer_by_name("docker")
    html = highlight("\n".join(dockerfile_commands), lexer, fmt)

    html = f"<style>{css_styles}</style><h1><code>modal.Image</code></h1>{html}"

    return HTML(html)


def get_image_dockerfile_commands(image):
    image_description = str(image)

    # dockerfile commands appear as a stringified Python list like below
    # Image(['CMD list', "FROM the modal image"])
    dockerfile_commands_list_str = image_description[len("Image([") : -len("])")]

    # we "unstringify" the list of strings before returning it
    dockerfile_commands = dockerfile_commands_list_str.split(", ")
    dockerfile_commands = [cmd.strip("'").strip('"') for cmd in dockerfile_commands]

    return dockerfile_commands
