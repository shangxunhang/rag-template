from pymilvus import MilvusClient

client = MilvusClient("D:/MyCode/rag-template/vector_store/test_milvus_lite.db")

print(client)
print(client.list_collections())