from rag_template.reader.json_reader import JsonReader
from rag_template.reader.jsonl_reader import JsonlReader
from rag_template.reader.txt_reader import TxtReader

READER_MAP = {
    ".txt": TxtReader,
    ".json": JsonReader,
    ".jsonl": JsonlReader,
}