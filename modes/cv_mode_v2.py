# modes/cv_mode_v2.py

import os
from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json

class CVModeV2(Mode):
    history: list[BaseMessage] = []

    def __init__(self, console: Console, cv_path: str, offer_path: str = "selected_offer.json", model: str = "llama3.2:1b", verbose: bool = False):
        super().__init__(console)
        self.cv_path = cv_path
        self.offer_path = offer_path
        self.model = model
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        p = subparser.add_parser(name, help="Générer des questions d’entretien à partir du CV et de l'offre")
        # nommage identique aux paramètres de __init__
        p.add_argument("cv_path", type=str, help="Chemin vers le fichier PDF du CV")
        p.add_argument("--offer_path", "-o", type=str, default="selected_offer.json", help="Chemin vers l'offre sélectionnée (défaut: selected_offer.json)")
        p.add_argument("--model", "-m", type=str, default="llama3.2:1b", help="Identifiant du modèle LLM")
        p.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")

    def run(self):
        # 1. Charger le CV et indexer en mémoire
        self.console.info(f"📄 Chargement du CV : {self.cv_path}")
        if not os.path.isfile(self.cv_path):
            self.console.error(f"CV introuvable : {self.cv_path}")
            return

        loader = PyPDFLoader(self.cv_path)
        splitter = SemanticChunker(embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest"))
        pages = list(loader.lazy_load())
        chunks = splitter.split_documents(pages)
        if self.verbose:
            self.console.info(f"{len(chunks)} chunks produits à partir du CV")

        cv_store = Chroma(embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"), persist_directory=None)
        cv_store.add_documents(chunks)

        # 2. Charger l'offre sélectionnée
        self.console.info(f"📄 Chargement de l'offre : {self.offer_path}")
        if not os.path.isfile(self.offer_path):
            self.console.error(f"Fichier d'offre introuvable : {self.offer_path}")
            return
        with open(self.offer_path, "r", encoding="utf-8") as f:
            offer = json.load(f)

        desc = offer.get("description", "")
        missions = offer.get("missions", [])
        competences = offer.get("compétences_requises", [])
        profil = offer.get("profil_recherché", "")
        offer_text = "\n".join([
            desc,
            "Missions: " + "; ".join(missions),
            "Compétences: " + "; ".join(competences),
            "Profil recherché: " + profil
        ])

        # 3. Récupération des meilleurs passages du CV
        query = f"Questions d'entretien pour le poste {offer.get('poste','')}"
        docs = cv_store.similarity_search(query, k=4)
        if self.verbose:
            self.console.info("Passages retenus pour le prompt:")
            for d in docs:
                self.console.info(d.page_content[:200] + "…")

        # 4. Construction du prompt et appel au LLM
        system_template = (
            "Tu es un recruteur expérimenté. Sur la base du CV et de l'offre ci-dessous, "
            "rédige des questions d'entretien adaptées.\n\n"
            "CV de l'utilisateur :\n{documents}\n\n"
            "Détails de l'offre :\n{offer}\n"
            "Les questions doivent couvrir : technique, comportemental, motivation."
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="messages"),
        ])
        model = init_chat_model(self.model, model_provider="ollama", temperature=0.7)
        chain = prompt | model | StrOutputParser()

        # 5. Génération et affichage dans la console
        self.history.append(HumanMessage(content="Génère des questions d'entretien adaptées."))
        self.console.bot_start()
        for chunk in chain.stream({
            "messages": self.history,
            "documents": docs,
            "offer": offer_text
        }):
            self.console.bot_chunk(chunk)
        self.console.bot_end()
        self.console.info("💬 Génération de questions terminée.")
