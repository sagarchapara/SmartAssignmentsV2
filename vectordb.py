import chromadb

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

def CreateCollection(collectionName):
    persist_directory = "assignment_db"

    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(name=collectionName)

    return collection


def Add(collection, user_id,  embedding, metadata=None, document = None):
    collection.add(
        ids = [user_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[document]
    )

def Get(collection, userIds):
    return collection.get(
        ids=userIds,
        include=["embeddings", "metadatas", "documents"]
    )