import os
import tempfile
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import qdrant_client

class Embedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()
        self.client = qdrant_client.QdrantClient(
    url="https://5d720bb4-0726-4fec-82df-394311676830.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="49ujY_YYOAfkBenRnr1LfuZ-_qkcnDq90nw1x9VoI4OVLlVgpGDQ4Q",
)

    def createEmbeddingsDir(self):
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def is_folder_empty(folder_path):
        return len(os.listdir(folder_path)) == 0

    def storeDocEmbeds(self, file, original_filename):

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
            
        def get_file_extension(uploaded_file):
            file_extension =  os.path.splitext(uploaded_file)[1].lower()
            
            return file_extension
        
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 2000,
                chunk_overlap  = 100,
                length_function = len,
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
            loader = CSVLoader(file_path=change_data,encoding="utf-8")
            data = loader.load()

        #UlXVk88UdjWzhPAhq695NDpXRbuLeJSHqOI6tkZX1XOSRL48EpklAw

        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split(text_splitter)
            
        embeddings = LlamaCppEmbeddings(model_path="model-q8_0.gguf")
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

        vectors = Qdrant(
            client=self.client, collection_name="dataset",
            embeddings=embeddings,
        )
        print("hellow")
        if vectors is None:
            print("fuck")
            vectors = Qdrant.from_documents(data, embeddings,
                            url="https://5d720bb4-0726-4fec-82df-394311676830.us-east4-0.gcp.cloud.qdrant.io:6333",
                            api_key="49ujY_YYOAfkBenRnr1LfuZ-_qkcnDq90nw1x9VoI4OVLlVgpGDQ4Q",
                            collection_name="dataset")
        os.remove(tmp_file_path)
        return vectors
        # Save the vectors to a pickle file
        #with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            #pickle.dump(vectors, f)

    def getDocEmbeds(self, file, original_filename):
        #if not os.path.isfile(f"{self.PATH}/storage.sqlite"):
        vectors = self.storeDocEmbeds(file, original_filename)
        return vectors
