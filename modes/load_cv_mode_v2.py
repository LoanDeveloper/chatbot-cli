# modes/load_cv_mode_v2.py

from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import os

class LoadCVModeV2(Mode):
    def __init__(self, console: Console, cv_path: str, offers_path: str = "offre.json", verbose: bool = False):
        super().__init__(console)
        self.cv_path = cv_path
        self.offers_path = offers_path
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        p = subparser.add_parser(name, help="Charger un CV et matcher des offres locales")
        # on nomme l’argument exactement cv_path pour coller au __init__
        p.add_argument(
            "cv_path",
            type=str,
            help="Chemin vers le fichier PDF du CV (ex: yael.pdf)"
        )
        # idem pour offers_path
        p.add_argument(
            "--offers_path", "-o",
            type=str,
            default="offre.json",
            help="Chemin vers le JSON des offres (défaut: offre.json)"
        )
        p.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Mode verbeux"
        )

    def run(self):
        # 1. Vérification et chargement du CV
        self.console.info(f"📄 Chargement du CV : {self.cv_path}")
        if not os.path.isfile(self.cv_path):
            self.console.error(f"CV introuvable : {self.cv_path}")
            return

        loader = PyPDFLoader(self.cv_path)
        splitter = SemanticChunker(embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest"))
        pages = list(loader.lazy_load())
        chunks = splitter.split_documents(pages)
        if self.verbose:
            for c in chunks:
                self.console.info(f"Chunk extrait ({len(c.page_content)} chars)")

        # 2. Chargement des offres locales
        if not os.path.isfile(self.offers_path):
            self.console.error(f"Fichier d'offres introuvable : {self.offers_path}")
            return
        with open(self.offers_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        offers = [data] if isinstance(data, dict) else data
        if self.verbose:
            self.console.info(f"{len(offers)} offre(s) chargée(s) depuis {self.offers_path}")

        # 3. Indexation des offres
        job_docs = []
        for o in offers:
            text = o.get("description", "")
            missions = o.get("missions", [])
            competences = o.get("compétences_requises", [])
            profil = o.get("profil_recherché", "")
            combined = "\n".join([
                text,
                "Missions: " + "; ".join(missions),
                "Compétences: " + "; ".join(competences),
                "Profil recherché: " + profil
            ])
            metadata = {
                "title": o.get("poste", ""),
                "company": o.get("entreprise", {}).get("nom", ""),
                "url": o.get("lien_de_candidature", "")
            }
            job_docs.append(Document(page_content=combined, metadata=metadata))

        job_store = Chroma(
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
            persist_directory=None
        )
        job_store.add_documents(job_docs)

        # 4. Similarité entre CV et offres
        full_text = "\n".join(p.page_content for p in pages)
        self.console.info("🔎 Recherche des offres les plus pertinentes…")
        results = job_store.similarity_search(full_text, k=min(5, len(job_docs)))

        # 5. Affichage et sauvegarde de l'offre sélectionnée
        self.console.info("✅ Offres recommandées :")
        for i, doc in enumerate(results, 1):
            md = doc.metadata
            self.console.info(f"{i}. {md['title']} @ {md['company']} → {md['url']}")

        # Enregistrer la meilleure offre pour la suite
        best = results[0]
        selected = None
        for doc, o in zip(job_docs, offers):
            if doc.metadata == best.metadata:
                selected = o
                break
        if selected:
            with open("selected_offer.json", "w", encoding="utf-8") as f:
                json.dump(selected, f, ensure_ascii=False, indent=2)
            self.console.info("💾 Offre sélectionnée enregistrée dans selected_offer.json")
