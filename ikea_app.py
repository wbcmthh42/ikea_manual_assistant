"""
IKEA Manual Assistant Application

This module implements a Streamlit-based interactive assistant for IKEA assembly manuals.
It combines multimodal AI capabilities to process both text and images from IKEA manuals,
providing intelligent responses to user queries about assembly instructions.

Key Features:
- PDF manual parsing with LlamaParse
- Multimodal processing of text and images
- Vector-based document retrieval
- Interactive chat interface with GPT-4o-mini
- Built-in PDF viewer for manual reference

Dependencies:
- OpenAI API (GPT-4o-mini)
- LlamaIndex
- Streamlit
- LlamaParse
- Various utility libraries (dotenv, nest_asyncio, etc.)

Environment Variables Required:
- OPENAI_API_KEY
- LLAMA_CLOUD_API_KEY

Usage:
Run the script using Streamlit:
    streamlit run ikea_app.py
"""

import os
import json
import re
from pathlib import Path
import typing as t
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from llama_index.core.schema import TextNode
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import ImageNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_parse import LlamaParse
import streamlit as st
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
import nest_asyncio

nest_asyncio.apply()
load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY

QA_PROMPT_TMPL = """\
You are a helpful IKEA assembly assistant that provides detailed guidance about IKEA product manuals.

Below is the context from the manual, including both images and text in two formats:
1. Markdown format (which includes tables and structured content)
2. Raw text format (which preserves the spatial layout)

Important Instructions:
1. Prioritize using information from images when available
2. Use the text/markdown information as supplementary context
3. Always mention the specific page number and document name where you found the information
4. If you cannot find the exact information in the provided context, clearly state that and suggest which manual section might be relevant
5. Provide step-by-step instructions when applicable
6. Include any relevant safety warnings or important notes from the manual

Context:
---------------------
{context_str}
---------------------

Based strictly on the provided context (not prior knowledge), please answer the following query.
If the information isn't available in the context, say so explicitly and mention which manual section might help.

Query: {query_str}
Answer: """

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

gpt_4o_mm = OpenAIMultiModal(model="gpt-4o")

# Configure streamlit UI page interface
st.set_page_config(
    page_title="IKEA Manual Assistant",
    layout="wide",  # This makes the app use full width
    initial_sidebar_state="expanded"
)

# Add custom CSS to reduce padding for streamlit UI
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .element-container {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

def get_data_files(data_dir) -> list[str]:
    """
    Retrieves all file paths from a specified directory.

    Args:
        data_dir (str): Path to the directory containing data files

    Returns:
        list[str]: List of full file paths for all files in the directory
    """
    files = []
    for f in os.listdir(data_dir):
        fname = os.path.join(data_dir, f)
        if os.path.isfile(fname):
            files.append(fname)
    return files

def get_page_number(file_name: str) -> int:
    """
    Extracts page number from image file names using regex.

    Args:
        file_name (str): Name of the image file (expected format: *-page_X.jpg)

    Returns:
        int: Page number if found in filename, 0 otherwise
    """
    match = re.search(r"-page_(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0


def _get_sorted_image_files(image_dir: str) -> List[Path]:
    """
    Retrieves and sorts image files by page number from specified directory.

    Args:
        image_dir (str): Path to directory containing image files

    Returns:
        list: Sorted list of Path objects for image files
    """
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files


def get_text_nodes(md_json_objs, image_dir) -> t.List[TextNode]:
    """
    Creates text nodes from JSON objects and associated images.

    Args:
        md_json_objs (list): List of JSON objects containing markdown content
        image_dir (str): Directory path containing associated images

    Returns:
        List[TextNode]: List of text nodes with metadata including image paths
    """
    nodes = []

    # Initialize the text splitter
    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=20
    )

    for result in md_json_objs:
        json_dicts = result["pages"]
        document_name = result["file_path"].split('/')[-1]

        docs = [doc["md"] for doc in json_dicts]  # extract text
        image_files = _get_sorted_image_files(image_dir)  # extract images

        for idx, doc in enumerate(docs):
            # Split the text into chunks
            chunks = text_splitter.split_text(doc)

            # Create a node for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                node = TextNode(
                    text=chunk,
                    metadata={
                        "image_path": str(image_files[idx]), 
                        "page_num": idx + 1, 
                        "document_name": document_name,
                        "chunk_idx": chunk_idx
                    },
                )
                nodes.append(node)

    return nodes

def create_index(nodes: List[TextNode]) -> Tuple[BaseRetriever, VectorStoreIndex, OpenAI]:
    """
    Creates or loads a vector store index for the provided nodes.

    Args:
        nodes (List[TextNode]): List of text nodes to index

    Returns:
        Tuple[BaseRetriever, VectorStoreIndex, OpenAI]: Tuple containing:
            - retriever: Index retriever
            - index: Vector store index
            - llm: Language model instance
    """
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    llm = OpenAI("gpt-4o-mini")

    Settings.llm = llm
    Settings.embed_model = embed_model

    if not os.path.exists("storage_manuals"):
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.storage_context.persist(persist_dir="./storage_manuals")
    else:
        ctx = StorageContext.from_defaults(persist_dir="./storage_manuals")
        index = load_index_from_storage(ctx)

    retriever = index.as_retriever()

    return retriever, index, llm



def main(file_dir: str) -> None:
    """
    Main application function that sets up the Streamlit interface and handles document processing.

    Args:
        file_dir (str): Directory path containing the manual files to process
    """
    st.title("IKEA Manual Assistant")
    if 'query_engine' not in st.session_state:
        global text_nodes
        DATA_DIR = file_dir

        parser = LlamaParse(
            result_type="markdown",
            parsing_instruction="You are given IKEA assembly instruction manuals",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            show_progress=True,
            verbose=True,
            invalidate_cache=True,
            do_not_cache=True,
            num_workers=8,
            language="en"
        )

        # Check if parsed files already exist
        if os.path.exists('parsed_data/md_json_objs.json') and os.path.exists('parsed_data/image_dicts.json'):
            # Load existing files
            with open('parsed_data/md_json_objs.json', 'r', encoding='utf-8') as f:
                md_json_objs = json.load(f)
            with open('parsed_data/image_dicts.json', 'r', encoding='utf-8') as f:
                image_dicts = json.load(f)
        else:
            # Create directory if it doesn't exist
            os.makedirs('parsed_data', exist_ok=True)

            # Parse new files
            files = get_data_files(DATA_DIR)

            md_json_objs = parser.get_json_result(files)
            image_dicts = parser.get_images(md_json_objs, download_path="data_images")

            # Save the parsed results
            with open('parsed_data/md_json_objs.json', 'w', encoding='utf-8') as f:
                json.dump(md_json_objs, f, ensure_ascii=False, indent=2)
            with open('parsed_data/image_dicts.json', 'w', encoding='utf-8') as f:
                json.dump(image_dicts, f, ensure_ascii=False, indent=2)

        text_nodes = get_text_nodes(md_json_objs, "data_images")
        retriever, index, llm = create_index(text_nodes)

        query_engine = MultimodalQueryEngine(
            qa_prompt=QA_PROMPT,
            retriever=index.as_retriever(similarity_top_k=9),
            multi_modal_llm=gpt_4o_mm,
            text_nodes=text_nodes,
            node_postprocessors=[],
        )

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="query_engine_tool",
            description="Useful for retrieving specific context from the data. Do NOT select if question asks for a summary of the data.",
            return_direct=True
        )
        st.session_state.agent = FunctionCallingAgentWorker.from_tools(
            [query_engine_tool],
            llm=llm,
            verbose=True,
            system_prompt="""You are an expert IKEA assembly assistant. When information is not found in the manual:
    1. Clearly state that the specific information is not available
    2. Suggest which pages to check
    3. Mention any related information that was found
    4. Offer to help with other assembly questions
    """
        ).as_agent()

    # Create the chat interface
    user_question = st.text_input("Ask a question about IKEA assembly:", key="user_input")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner('Finding answer...'):
                agent_response = st.session_state.agent.chat(user_question)
                st.write("Answer:")
                st.write(agent_response.response)  # Display the text response

        else:
            st.warning("Please enter a question!")


class MultimodalQueryEngine(CustomQueryEngine):
    """
    Custom query engine for handling multimodal (text + image) queries.

    Attributes:
        qa_prompt (PromptTemplate): Template for question-answering
        retriever (BaseRetriever): Retriever for finding relevant nodes
        multi_modal_llm (OpenAIMultiModal): Multimodal language model
        text_nodes (List[TextNode]): List of text nodes from the document
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Post-processing steps for nodes
    """
    qa_prompt: PromptTemplate
    retriever: BaseRetriever
    multi_modal_llm: OpenAIMultiModal
    node_postprocessors: Optional[List[BaseNodePostprocessor]]
    text_nodes: List[TextNode]

    def __init__(
        self,
        qa_prompt: PromptTemplate,
        retriever: BaseRetriever,
        multi_modal_llm: OpenAIMultiModal,
        text_nodes: List[TextNode],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = [],
    ):
        super().__init__(
            qa_prompt=qa_prompt,
            retriever=retriever,
            multi_modal_llm=multi_modal_llm,
            node_postprocessors=node_postprocessors,
            text_nodes=text_nodes,
        )

    def custom_query(self, query_str: str):
        """
        Processes a query using both text and image content.

        Args:
            query_str (str): The query string from the user

        Returns:
            Response: Response object containing the answer and source nodes
        """
        nodes = self.retriever.retrieve(query_str)

        if not nodes:
            return Response(
                response="I couldn't find any relevant information in the provided manuals. Please check if you have uploaded the correct manual or try rephrasing your question.",
                source_nodes=[],
                metadata={"text_nodes": [], "image_nodes": []}
            )

        # Get page range from available nodes
        page_numbers = [n.node.metadata.get("page_num", 0) for n in nodes]
        min_page = min(page_numbers)
        max_page = max(page_numbers)

        # create image nodes from the image associated with those nodes
        image_nodes = [
            NodeWithScore(node=ImageNode(image_path=n.node.metadata["image_path"]))
            for n in nodes
        ]

        # create context string from parsed markdown text
        ctx_str = "\n\n".join(
            [r.node.get_content(metadata_mode=MetadataMode.LLM).strip() for r in nodes]
        )

        # If content is found but might be incomplete, include page range suggestion
        if not any(keyword in ctx_str.lower() for keyword in [str(query_str)]):
            document_names = set(n.node.metadata.get("document_name", "") for n in nodes)
            suggestion = "\n\nNote: The requested information was not found in the current context. "
            suggestion += f"You may want to check pages {min_page-3} to {max_page+3} of {', '.join(document_names)}. "
            suggestion += "The available context contains information about other assembly steps and parts. "
            suggestion += "Please feel free to ask about other aspects of the assembly process."
            ctx_str += suggestion

        # prompt for the LLM
        fmt_prompt = self.qa_prompt.format(context_str=ctx_str, query_str=query_str)

        # use the multimodal LLM to interpret images and generate a response to the prompt
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[image_node.node for image_node in image_nodes],
        )
        return Response(
            response=str(llm_response),
            source_nodes=nodes,
            metadata={"text_nodes": self.text_nodes, "image_nodes": image_nodes},
        )

if __name__ == "__main__":
    # Set default file directory
    if 'file_dir' not in st.session_state:
        st.session_state.file_dir = "./manuals/files"

    # Allow user to change directory
    file_dir = st.text_input("Enter PDF file path:",
                            value=st.session_state.file_dir,
                            key="file_dir_input")
    main(file_dir=file_dir)

    with st.sidebar:
        st.header("Manual Viewer")
        # Declare variables
        if 'pdf_ref' not in ss:
            ss.pdf_ref = None

        # Get list of PDFs from the files directory
        pdf_files = [f for f in os.listdir(file_dir) if f.endswith('.pdf')]

        # Create dropdown for PDF selection
        selected_pdf = st.selectbox(
            "Select PDF Manual",
            options=[""] + pdf_files,  # Empty option + list of PDFs
            index=0  # Default to first option (empty)
        )

        # Handle PDF selection/upload
        if selected_pdf:
            with open(os.path.join(file_dir, selected_pdf), 'rb') as file:
                binary_data = file.read()
                ss.pdf_ref = selected_pdf

        # Display PDF if we have a reference
        if ss.pdf_ref:
            pdf_viewer(input=binary_data, width=400)  # Reduced width for sidebar
