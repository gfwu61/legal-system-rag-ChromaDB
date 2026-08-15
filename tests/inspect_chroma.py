import chromadb


DB_PATH = "chroma_legal_rag"
COLLECTION_NAME = "langchain"


# Connect to the new Chroma database
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)


# Read documents and metadata
data = collection.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]
metadatas = data["metadatas"]


print(f"Database: {DB_PATH}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Number of documents: {len(documents)}")
print()


# ------------------------------------------------------------
# Check paragraph 573c
# ------------------------------------------------------------

for index, metadata in enumerate(metadatas):

    if metadata.get("paragraph") == "573c":

        print("=" * 80)
        print("INDEX:", index)
        print("METADATA:", metadata)
        print()
        print("CONTENT:")
        print(documents[index])