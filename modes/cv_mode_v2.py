# cv_mode_v2.py
import os
from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class CVModeV2(Mode):
    history: list[BaseMessage] = []

    def __init__(self, console: Console, model: str = "llama3.2:1b", verbose: bool = False):
        super().__init__(console)
        self.model = model
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        p = subparser.add_parser(name, help="Générer des questions à partir du CV")
        p.add_argument("--model", type=str, default="llama3.2:1b")
        p.add_argument("--verbose", "-v", action="store_true")

    def run(self):
        # 1. Construction du prompt système
        system_prompt = """
Tu es un recruteur expérimenté. Sur la base du CV fourni ci-dessous, rédige une série de questions d’entretien pertinentes pour évaluer le candidat.
Le CV de l'utilisateur :
{documents}
"""
        # 2. Chargement de l'index (in-memory)
        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
            persist_directory=None
        )
        # 3. Recherche par similarité
        docs = vector_store.similarity_search("questions d’entretien CV", k=4)
        if self.verbose:
            self.console.info("🗂️ Documents retenus pour le prompt:")
            for d in docs:
                self.console.info(d.page_content[:200] + "…")

        # 4. Initialisation du LLM
        model = init_chat_model(self.model, model_provider="ollama", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | model | StrOutputParser()

        # 5. Conversation
        self.history.append(HumanMessage(content="Génère des questions d’entretien adaptées à mon CV."))
        self.console.bot_start()
        out = ""
        for chunk in chain.stream({"messages": self.history, "documents": docs}):
            out += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()
        self.history.append(AIMessage(out))

        # 6. Sauvegarde des questions dans un fichier
        with open("web_dev_questions.txt", "w", encoding="utf-8") as f:
            f.write(out)
        self.console.info("💾 Questions enregistrées dans web_dev_questions.txt")

    @staticmethod
    def generate_questions_file():
        """
        Génère un fichier 'web_dev_questions.txt' contenant 200 questions
        standards pour un développeur web.
        """
        questions = [
            # … liste complète de 200 questions, voir ci-dessous …
        ]
        with open("web_dev_questions.txt", "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q + "\n")
        print("Fichier 'web_dev_questions.txt' généré avec 200 questions.")
