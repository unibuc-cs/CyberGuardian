
import etlUtils.etl_shared
import json
from tqdm import tqdm
import os

def add_pdf_to_local_db(documents_json):
    import json

    res = []
    for d in documents_json:
        doc_selected = {#'_id' : str(d['_id']),
        'type' : d['type'],
        'text' : d['text'],
        'title' : d['metadata']['title'],
        'source' : d['metadata']['source'],
        'page' : d['metadata']['page'],
        'data' : str(d['metadata'].get('date', "2023"))}

        res.append(json.dumps(doc_selected)+ '\n')
    return res

def transform_papers_to_json(papers_path, demoMode=False):
    with open(papers_path) as f:
        pdf_infos = json.load(f)

    if demoMode:
        pdf_infos = etlUtils.etl_shared.subsampleForDemo(pdf_infos, etlUtils.etl_shared.DEMO_PDF_SAMPLES)

    ## DEBUG
    #pdf_infos = pdf_infos[:10]

    # print(pdf_infos[:100:20])

    all_papers_texts = []

    # Augment the paper data by finding direct PDF URLs where possible
    paper_data = pdf_infos
    if etlUtils.etl_shared.PARALLEL_USE:
        from multiprocessing import Pool, TimeoutError
        n_processes = etlUtils.etl_shared.PARALLEL_NUM_PROCESSES
        with Pool(processes=n_processes) as pool:
            results = etlUtils.etl_shared.imap_unordered_bar(get_pdf_url, paper_data,
                                                            n_processes=n_processes,
                                                            desc="Parallel getting url for pdfs: ")
        paper_data = results

        # turn the PDFs into JSON documents
        with Pool(processes=n_processes) as pool:
            results = etlUtils.etl_shared.imap_unordered_bar(extract_pdf, paper_data,
                                                             n_processes=n_processes,
                                                             desc="Parallel extraction of papers content: ")

        for paper_text in results:
            all_papers_texts.append(paper_text)
    else:

        for pdf_i in tqdm(paper_data, "Serial getting papers URLs", total=len(paper_data)):
            get_pdf_url(pdf_i)

        # turn the PDFs into JSON documents
        for paper in tqdm(paper_data, desc="Serial extraction of papers content: "):
            paper_text = extract_pdf(paper)
            if paper_text is None or len(paper_text) == 0:
                continue
            all_papers_texts.append(paper_text)


    documents = etlUtils.etl_shared.unchunk(all_papers_texts)

    # Test out debug
    #pp.pprint(documents[0]["metadata"])

    # Split document list into 10 pieces
    n_chunks = 10
    chunked_documents = etlUtils.etl_shared.chunk_into(documents, n_chunks)


    # Document database online store
    results = []
    jsonlines = []
    for mydoc in tqdm(chunked_documents, total=n_chunks, desc="Writing chunks to mongodb and setting up local pdf database"):
        res = etlUtils.etl_shared.add_to_mongo_db(mydoc)
        res_json = add_pdf_to_local_db(mydoc)

        results.append(res)
        jsonlines.append(res_json)

    jsonlines = etlUtils.etl_shared.unchunk(jsonlines)

    with open(os.environ["PDF_LOCAL_JSON_DB"], 'w') as the_file:
        the_file.writelines(jsonlines)


    # Pull only arxiv other_papers
    """
    query = {"metadata.source": {"$regex": "arxiv\.org", "$options": "i"}}
    # Project out the text field, it can get large
    projection = {"text": 0}
    # get just one result to show it worked
    result = docstore.query_one(query, projection)
    pp.pprint(result)
    """


# Extracts the text from a PDF and adds metadata.
def extract_pdf(paper_data):
    import logging
    import arxiv
    from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader


    #### DEBUG ONLY !!!
    print(paper_data["url"])
    if 'url' in paper_data and ("2311.16062v1" not in paper_data['url']):
        return []


    pdf_url = None
    if "folder" in paper_data:
        loader = PyPDFDirectoryLoader(paper_data["folder"])
        #docs = loader.load_and_split()
    elif "singlepdf" in paper_data:
        loader = PyPDFLoader(paper_data['singlepdf'])
    else:
        pdf_url = paper_data.get("pdf_url")
        if pdf_url is None:
            return []

        logger = logging.getLogger("pypdf")
        logger.setLevel(logging.ERROR)

        try:
            loader = PyPDFLoader(pdf_url)
        except Exception as e:
            print(e)
            return e

    try:
        documents = loader.load_and_split()
    except Exception as e:
        print(e)
        return []

    documents = [document.dict() for document in documents]
    invalid_toomuch_unicode_documents = []
    for document in documents:  # rename page_content to text, handle non-unicode data
        document["text"] = (
            document["page_content"].encode("utf-8", errors="replace").decode()
        )
        str_to_ascii = document["page_content"].encode("ascii", "ignore")
        ascii_count_on_page = len(str_to_ascii)
        unicode_count_on_page = len(document["text"])
        ration_ascii_over_unicode = float(ascii_count_on_page / unicode_count_on_page)
        if ration_ascii_over_unicode < 0.80:
            print(f"INVALID ASCII COUNT ON PAGE: ration:{ration_ascii_over_unicode}. ascii:{ascii_count_on_page}. Source: {document['metadata']['source']}")
            invalid_toomuch_unicode_documents.append(document)
        document.pop("page_content")

    for document in invalid_toomuch_unicode_documents:
        documents.remove(document)

    if pdf_url is not None:
        if "arxiv" in pdf_url:
            arxiv_id = extract_arxiv_id_from_url(pdf_url)
            # create an arXiV database client with a 5 second delay between requests
            client = arxiv.Client(page_size=1, delay_seconds=5, num_retries=5)
            # describe a search of arXiV's database
            search_query = arxiv.Search(id_list=[arxiv_id], max_results=1)
            try:
                # execute the search with the client and get the first result
                result = next(client.results(search_query))
            except ConnectionResetError as e:
                print("Triggered request limit on arxiv.org, retrying", e)
                return []
            except Exception as e:
                print(e)
                return []

            metadata = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "date": result.updated,
            }
        else:
            metadata = {"title": paper_data.get("title")}
    else:
        metadata = {"title": "notknown"}

    documents = annotate_endmatter(documents)

    for document in documents:
        document["metadata"]["source"] = paper_data.get("url", pdf_url)
        document["metadata"] |= metadata
        title, page = (
            document["metadata"]["title"],
            document["metadata"]["page"],
        )
        if title:
            document["metadata"]["full-title"] = f"{title} - p{page}"

    documents = etlUtils.etl_shared.add_metadata(documents)

    return documents

def concatenate_content_multiple_pages(documents, ignore_endmatter=True):
    return "".join(doc['text'] for doc in documents if doc['metadata'].get("is_endmatter", False) is False)


# Attempts to extract a PDF URL from a paper's URL.
def get_pdf_url(paper_data):

    if "folder" in paper_data:
        return paper_data

    url = paper_data["url"]
    if url.strip("#/").endswith(".pdf"):
        pdf_url = url
    elif "arxiv.org" in url:
        arxiv_id = extract_arxiv_id_from_url(url)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    elif "aclanthology.org" in url:
        pdf_url = url.strip("/")
        url += ".pdf"
    else:
        pdf_url = None
    paper_data["pdf_url"] = pdf_url

    return paper_data


def annotate_endmatter(pages, min_pages=3):
    """Heuristic for detecting reference sections."""
    out, after_references = [], False
    for idx, page in enumerate(pages):

        # Starting a new chunk ? Then remove after_references
        if after_references is True and ('metadata' in page and 'page' in page['metadata']):
            if page['metadata']['page'] == 0:
                after_references = False

        content = page["text"].lower()
        if idx >= min_pages and ("references" in content or "bibliography" in content):
            after_references = True
        page["metadata"]["is_endmatter"] = after_references
        out.append(page)
    return out


def extract_arxiv_id_from_url(url):
    import re

    # pattern = r"(?:arxiv\.org/abs/|arxiv\.org/pdf/)(\d{4}\.\d{4,5}(?:v\d+)?)"
    match_arxiv_url = r"(?:arxiv\.org/abs/|arxiv\.org/pdf/)"
    match_id = r"(\d{4}\.\d{4,5}(?:v\d+)?)"  # 4 digits, a dot, and 4 or 5 digits
    optional_version = r"(?:v\d+)?"

    pattern = match_arxiv_url + match_id + optional_version

    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


# Calls the ETL pipeline using a JSON file with PDF metadata.
"""
def main(json_path, collection=None, db=None):
    import json
    from pathlib import Path

    json_path = Path(json_path).resolve()

    if not json_path.exists():
        print(f"{json_path} not found, writing to it from the database.")
        paper_data = fetch_papers.call()
        paper_data_json = json.dumps(paper_data, indent=2)
        with open(json_path, "w") as f:
            f.write(paper_data_json)

    with open(json_path) as f:
        paper_data = json.load(f)

    paper_data = get_pdf_url.map(paper_data, return_exceptions=True)

    documents = etlUtils.etl_shared.unchunk(extract_pdf.map(paper_data, return_exceptions=True))

    with etlUtils.etl_shared.stub.run():
        chunked_documents = etlUtils.etl_shared.chunk_into(documents, 10)
        list(
            etlUtils.etl_shared.add_to_mongo_db.map(
                chunked_documents, kwargs={"db": db, "collection": collection}
            )
        )
"""
