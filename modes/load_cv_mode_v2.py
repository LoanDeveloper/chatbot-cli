# load_cv_mode_v2.py
from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import requests

class LoadCVModeV2(Mode):
    def __init__(self, console: Console, cv_path: str, verbose: bool = False):
        super().__init__(console)
        self.cv_path = cv_path
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        p = subparser.add_parser(name, help="Charger un CV et matcher des offres")
        p.add_argument("cv", type=str, help="Chemin vers le fichier PDF du CV")
        p.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")

    def run(self):
        self.console.info(f"📄 Chargement du CV : {self.cv_path}")
        # 1. Extraction et segmentation
        loader = PyPDFLoader(self.cv_path)
        splitter = SemanticChunker(
            embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
        )
        pages = list(loader.lazy_load())
        chunks = splitter.split_documents(pages)
        if self.verbose:
            for c in chunks:
                self.console.info(f"Chunk ({len(c.page_content)} chars)")
        # 2. Indexation dans Chroma (in-memory, pas de persist pour ne rien stocker)
        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
            persist_directory=None
        )
        vector_store.add_documents(chunks)
        # 3. Recherche d'offres correspondantes
        full_text = "\n".join([p.page_content for p in pages])
        offers = self.fetch_job_offers(full_text)
        self.console.info("🔎 Offres d'emploi trouvées :")
        for o in offers:
            self.console.info(f"- {o['title']} @ {o['company']} ({o['url']})")

    def fetch_job_offers(self, cv_text: str) -> list[dict]:
        """
        Appel fictif à une API de job matching. 
        On envoie le texte du CV et on récupère une liste d'offres.
        """
        # Exemple : remplacer par un vrai endpoint
        resp = requests.post("https://api.monsite/jobs/match", json={"cv": cv_text})
        if resp.status_code == 200:
            return resp.json().get("offers", [])
        return []
