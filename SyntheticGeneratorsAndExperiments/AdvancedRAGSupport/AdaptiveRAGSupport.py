from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Union, List, Dict, Any, NoReturn
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer
from time import time
from langchain_huggingface import HuggingFacePipeline
import torch
import advancedRagUtils
from advancedRagUtils import logger
from langchain_community.document_loaders import PyPDFLoader
from RAGPromptsFactory import RAGPromptsFactory

"""
Combine ideas from paper RAG papers into a RAG agent:

- **Routing:**  Adaptive RAG ([paper](https://arxiv.org/abs/2403.14403)). Route questions to different retrieval approaches
- **Fallback:** Corrective RAG ([paper](https://arxiv.org/pdf/2401.15884.pdf)). Fallback to web search if docs are not relevant to query
- **Self-correction:** Self-RAG ([paper](https://arxiv.org/abs/2310.11511)). Fix answers w/ hallucinations or don’t address question

See picture from langchain to see the overview of the demo impl. of the AdaptiveRAGSupport class
"""


# The AdaptiveRAGSupport class
class AdaptiveRAGSupport:
    # The maximum number of documents returned by the retriever
    DEFAULT_MAX_DOCS_RETURNED = 3
    NUM_WEB_SEARCH_RESULTS = 3

    # The default model id if none is provided in the constructor
    DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __init__(self, model: Union[HuggingFacePipeline, None],
                tokenizer: Union[AutoTokenizer, None],
                    retriever: Union[VectorStoreRetriever, None]):
        # Init model if none is provided
        self.llm_pipeline = None
        self.graderLLMTool = None

        env_model_things = [model, tokenizer, retriever]
        if all(env_model_things):
            self.model = model
            self.tokenizer = tokenizer
            self.retriever = retriever
        else:
            assert any(env_model_things) == False, "Either all or none of the model, tokenizer and retriever should be provided"
            self.create_env_model_things()
            self.create_retriever()

        self.prompts_factory = RAGPromptsFactory(self.tokenizer)

    # Create the LLM model and pipeline
    def create_env_model_things(self):
        # Load the LLM model
        model_id = self.DEFAULT_MODEL_ID
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        time_start = time()
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            # max_new_tokens=1024
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
        )

        time_end = time()
        print(f"Prepare model, tokenizer: {round(time_end - time_start, 3)} sec.")

        tokenizer = AutoTokenizer.from_pretrained(model_id,skip_special_tokens=True, trust_remote_code=True)

        # Create a query pipeline using transformers and test it!
        time_start = time()
        query_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            # max_length=1024,
            device_map="auto",
            eos_token_id=tokenizer.eos_token_id,
            truncation=True,
            return_full_text=False)
        time_end = time()
        logger.info(f"Prepare pipeline: {round(time_end - time_start, 3)} sec.")

        self.model = model
        self.model.generation_config.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        # Create the LLM pipeline
        self.llm_pipeline = HuggingFacePipeline(pipeline=query_pipeline)#, pipeline_kwargs={'skip_prompt':True})

        assert self.model is not None and self.tokenizer is not None, "The model should not be None"

    @staticmethod
    # Colorize the text for better readability in the notebook
    def print_colorized_text(text):
        for word, color in zip(["Reasoning", "Question", "Answer", "Total time"], ["blue", "red", "green", "magenta"]):
            text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")

        from IPython.display import display, Markdown
        display(Markdown(text))


    # Test the model and pipeline
    def test_llm_pipeline(self, message: str, skip_prompt: bool):
        """
        Perform a query
        """
        time_start = time()
        sequences = self.llm_pipeline(message, skip_prompt=skip_prompt)
        time_end = time()
        total_time = f"{round(time_end - time_start, 3)} sec."

        #question = sequences[0]['generated_text'][:len(message)]
        # answer = sequences[0]['generated_text'][len(message):]

        return f"Question: {message}\nAnswer: {sequences}\nTotal time: {total_time}"

    # Create the retriever
    def create_retriever(self):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.vectorstores import Chroma
        from langchain_nomic.embeddings import NomicEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        embedding = NomicEmbeddings(model="nomic-embed-text-v1.5",
                                    inference_mode='local')


        # Try to use cache first if it exists
        persist_directory = './RagSupport_test'
        vectorstore = Chroma(collection_name="rag-chroma",
                             persist_directory=persist_directory,
                             embedding_function=embedding,)
        retriever = None
        if vectorstore._collection.count() > 0:
            retriever = vectorstore.as_retriever()
        else:
            urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
            ]

            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # Add to vectorDB
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=embedding,
                persist_directory = persist_directory
            )
            retriever = vectorstore.as_retriever()

        assert retriever is not None, "The retriever should not be None"
        self.retriever = retriever

    # Get the documents from the retriever closest to the query
    def get_docs_content_by_query(self, query: str, num_max_docs_returned: int, format_as_single_text: bool, verbose: bool = False) -> str:
        """
        Args:
            query: the query
            verbose: whether to print the retrieved documents
            format_as_single_text: whether to format the documents as a single text or return a list of Documents objects
            num_max_docs_returned: the maximum number of documents to return
        Returns:
            Str
        """
        docs_list = self.retriever.invoke(query, verbose=verbose, kwargs={'k': num_max_docs_returned})

        def format_docs(docs) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        docs_list = self.retriever.invoke(query, verbose=verbose)
        
        docs_content = docs_list
        if format_as_single_text:
            docs_content = format_docs(docs_list)
            
        return docs_content

    # Answer a question with RAG
    def answer_with_rag(self, /, question: str, docs_content: str, skip_prompt=True) -> str:
        """
        Args:
            question: the question
            docs_content: the documents content as context
            skip_prompt: whether to skip the prompt
        Returns:
            Str
        """
        rag_prompt = self.prompts_factory.rag_prompt
        # prompt_value = rag_prompt.format(question=question, context=format_docs(docs_content))

        rag_chain = rag_prompt | self.llm_pipeline.bind(skip_prompt=skip_prompt) | StrOutputParser()
        res = rag_chain.invoke({"question": question, "context": docs_content})
        assert type(res) == str, f"Expected a string but got {type(res)}"
        return res

    # Checks if the answer is useful to resolve the question. JSON output by the answer grader
    def answer_grader(self, question: str, answer: str, skip_prompt=True) -> str:
        """
        Args:
            question: the question
            answer: the answer
            skip_prompt: whether to skip the prompt
        Returns:
            Str
        """
        answer_grader = self.prompts_factory.answer_grader_prompt | self.llm_pipeline.bind(
            skip_prompt=skip_prompt) | JsonOutputParser()
        return answer_grader.invoke({"question": question, "generation": answer})

    # Checks if the answer is hallucinated or sustained by the documents. JSON output by the hallucination grader
    def hallucination_grader(self, answer: str, docs_content: str, skip_prompt=True) -> str:
        """
        Args:
            answer: the answer
            docs_content: the documents content as context
            skip_prompt: whether to skip the prompt
        Returns:
            str
        """
        hallucination_grader = self.prompts_factory.hallucination_grader_prompt | self.llm_pipeline.bind(
            skip_prompt=skip_prompt) | JsonOutputParser()
        res = hallucination_grader.invoke({"documents": docs_content, "generation": answer})
        return res

    def question_router(self, question: str, skip_prompt=True) -> Dict[str, Any]:
        """
        Args:
            question: the question
            skip_prompt: whether to skip the prompt
        Returns:
            Dict[str, Any]
        """
        
        question_router = self.prompts_factory.question_router_prompt | self.llm_pipeline.bind(
            skip_prompt=skip_prompt) | JsonOutputParser()
        res = question_router.invoke({"question": question, "expert_domains_text": "llm agents, agent memory, prompt engineering, adversarial attacks"})
        return res
    

