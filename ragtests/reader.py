

from rag_template.configs.RAGConfig import *
from rag_template.chunker.chunk import chunk_documents
from rag_template.cleaner.clean import clean_documents
from rag_template.reader.reader_factory import load_documents
from rag_template.reader.txt_reader import *

if __name__ == '__main__':



    document=load_documents(RAW_DATA_DIR)
    print(document)
