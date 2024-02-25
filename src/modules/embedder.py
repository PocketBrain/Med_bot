import os
import tempfile
import pandas as pd
import pickle
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Qdrant, FAISS
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from qdrant_client import QdrantClient



class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()
        self.embeddings = LlamaCppEmbeddings(model_path="model-q8_0.gguf")
        #self.CLIENT = QdrantClient("localhost", port=6333)

    def createEmbeddingsDir(self):
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def is_file_exists(self, file_name):
        return os.path.exists(self.EMBEDPATH + file_name + "storage.sqlite")

    def storeDocEmbeds(self, file, original_filename):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        def get_file_extension(uploaded_file):
            file_extension = os.path.splitext(uploaded_file)[1].lower()

            return file_extension

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len,
        )

        file_extension = get_file_extension(original_filename)

        if file_extension == ".csv":
            df = pd.read_csv(tmp_file_path, sep=";")
            loader = DataFrameLoader(df, page_content_column='question')
            data = loader.load()

        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)  
            data = loader.load_and_split(text_splitter)

        elif file_extension == ".xlsx":
            change_data = pd.read_excel(tmp_file_path, engine='openpyxl')
            change_data.to_csv(tmp_file_path, index=False)
            loader = CSVLoader(file_path=change_data, encoding="utf-8")
            data = loader.load()


        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split(text_splitter)

        print("hellow")
        vectors = FAISS.from_documents(data, self.embeddings)
        os.remove(tmp_file_path)

        vectors.save_local(f"embeddings/{original_filename}")


        return vectors
        # Save the vectors to a pickle file
        #with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            #pickle.dump(vectors, f)

    def getDocEmbeds(self, file, original_filename):
        if not os.path.isfile(f"{self.PATH}/{original_filename}/index.pkl"):
            print("пиздец")
            vectors = self.storeDocEmbeds(file, original_filename)
        else:
            print("Но кок")
            vectors = FAISS.load_local(f"{self.PATH}/{original_filename}", self.embeddings)

        return vectors
