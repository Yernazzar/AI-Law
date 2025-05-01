import os
import json
import hashlib
import shutil
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain.docstore.document import Document
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.base import Embeddings
import docx
import pdfplumber
import chardet
from razdel import sentenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import telebot
import logging
from simulation import SimulationManager, SimulationState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    warnings.warn("OPENAI_API_KEY not found in .env file")

# Telegram Token
TELEGRAM_BOT_TOKEN = "8065137762:AAGJ8a-yDczf8L2b54nzR--nwM-0iKELEPc"

# Folders setup
DATA_FOLDER = os.getenv("DATA_FOLDER", "legal_data")
INDEXES_FOLDER = os.getenv("INDEXES_FOLDER", "legal_indexes")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(INDEXES_FOLDER, exist_ok=True)


# Embeddings class for text
class MyEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# Document manager class
class DocumentManager:
    def __init__(self, data_folder, metadata_file="legal_metadata.json"):
        self.data_folder = data_folder
        self.metadata_file = os.path.join(data_folder, metadata_file)
        self.ensure_metadata_exists()

    def ensure_metadata_exists(self):
        os.makedirs(self.data_folder, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump({"documents": []}, f, ensure_ascii=False, indent=2)

    def load_metadata(self):
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_metadata(self, metadata):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def calculate_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def extract_full_text(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.docx':
                doc = docx.Document(file_path)
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            elif file_ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif file_ext == '.txt':
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    detected_encoding = chardet.detect(raw_data)['encoding']
                with open(file_path, 'r', encoding=detected_encoding or 'utf-8', errors='replace') as file:
                    return file.read()
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

    def get_active_documents(self):
        metadata = self.load_metadata()
        return [doc for doc in metadata.get("documents", []) if doc.get("status") == "active"]

    def get_document_path(self, file_name):
        """Find the full path of a document by its filename"""
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file == file_name:
                    return os.path.join(root, file)
        return None


# Document processing functions
def split_text_into_sentences(text):
    return [s.text for s in sentenize(text)]


def create_chunks_by_sentence(text, file_metadata, target_chunk_size=512, min_chunk_size=256, overlap_size=50):
    sentences = split_text_into_sentences(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > target_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            if overlap_size > 0:
                overlap_sentences = []
                overlap_tokens = 0
                j = len(current_chunk) - 1
                while j >= 0 and overlap_tokens < overlap_size:
                    overlap_sentences.insert(0, current_chunk[j])
                    overlap_tokens += len(current_chunk[j].split())
                    j -= 1
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk = [sentence]
                current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunk_dicts = []
    for idx, chunk in enumerate(chunks):
        if len(chunk.split()) >= min_chunk_size:
            data = {
                "id": f"{file_metadata.get('file_name', 'unknown')}-chunk-{idx}",
                "text": chunk,
                "metadata": {**file_metadata, "chunk_id": idx, "token_count": len(chunk.split())}
            }
            chunk_dicts.append(data)
    return chunk_dicts


def process_document(file_path, target_chunk_size=512, min_chunk_size=256, overlap_size=50):
    file_extension = os.path.splitext(file_path)[1].lower()
    file_metadata = {"file_name": os.path.basename(file_path), "file_path": file_path}
    text = ""

    if file_extension == '.docx':
        doc = docx.Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        file_metadata["file_type"] = "docx"
    elif file_extension == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            file_metadata["file_type"] = "pdf"
            file_metadata["num_pages"] = len(pdf.pages)
    elif file_extension == '.txt':
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)['encoding']
        with open(file_path, 'r', encoding=detected_encoding or 'utf-8', errors='replace') as file:
            text = file.read()
        file_metadata["file_type"] = "txt"
    else:
        logger.info(f"Unsupported file type: {file_path}")
        return []

    if len(text.strip()) < 50:
        logger.info(f"File {file_metadata['file_name']} contains less than 50 characters.")
        return []
    return create_chunks_by_sentence(text, file_metadata, target_chunk_size, min_chunk_size, overlap_size)


def process_document_folder(folder_path, **kwargs):
    all_chunks = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.docx', '.pdf', '.txt')) and not file.startswith('.'):
                file_path = os.path.join(root, file)
                if file != "legal_metadata.json":
                    logger.debug(f"Processing file: {file_path}")
                    chunks = process_document(file_path, **kwargs)
                    all_chunks.extend(chunks)
    return all_chunks


# Vector storage functions
def create_faiss_index(chunks, embeddings_model):
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        vector_size = embeddings_model.embed_query("Empty index").shape[0]
        index = faiss.IndexFlatL2(vector_size)
        return index, []

    embed_array = np.array(embeddings_model.embed_documents(texts)).astype("float32")
    vector_size = embed_array.shape[1]
    index = faiss.IndexFlatL2(vector_size)
    index.add(embed_array)
    return index, embed_array


def save_faiss_index(index, metadata, index_path, metadata_path):
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_or_rebuild_vectorstore(data_folder, indexes_folder, embeddings_model):
    os.makedirs(indexes_folder, exist_ok=True)
    fingerprint_file = os.path.join(indexes_folder, "index_fingerprint.json")
    index_path = os.path.join(indexes_folder, "index.faiss")
    metadata_path = os.path.join(indexes_folder, "document_metadata.json")

    current_fingerprint = {}
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.docx', '.pdf', '.txt')) and not file.startswith('.'):
                if file != "legal_metadata.json":
                    path = os.path.join(root, file)
                    current_fingerprint[path] = os.path.getmtime(path)

    fingerprint_hash = hashlib.md5(json.dumps(current_fingerprint, sort_keys=True).encode()).hexdigest()

    previous_fingerprint = None
    if os.path.exists(fingerprint_file):
        with open(fingerprint_file, 'r') as f:
            try:
                previous_fingerprint = json.load(f)
            except json.JSONDecodeError:
                previous_fingerprint = None

    if (os.path.exists(index_path) and os.path.exists(metadata_path) and
            previous_fingerprint == fingerprint_hash):
        try:
            logger.info("Loading existing vector store...")
            vectorstore = LC_FAISS.load_local(indexes_folder, embeddings_model, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

    logger.info("Recreating vector index...")
    chunks = process_document_folder(data_folder, target_chunk_size=512, min_chunk_size=256, overlap_size=150)

    if not chunks:
        logger.info("No documents found for indexing. Creating empty index.")
        vectorstore = LC_FAISS.from_documents(
            [Document(page_content="Empty index", metadata={"source": "empty"})],
            embeddings_model
        )
        vectorstore.save_local(indexes_folder)
        with open(fingerprint_file, 'w') as f:
            json.dump(fingerprint_hash, f)
        return vectorstore

    docs = [Document(page_content=ch["text"], metadata=ch["metadata"]) for ch in chunks]
    vectorstore = LC_FAISS.from_documents(docs, embeddings_model)
    vectorstore.save_local(indexes_folder)
    with open(fingerprint_file, 'w') as f:
        json.dump(fingerprint_hash, f)
    return vectorstore


# Prompt template
def get_legal_prompt_template():
    return (
        "Вы — ассистент по юридическим консультациям для жителей, называетесь ThemisBot. "
        "Используйте предоставленный контекст для точного и краткого ответа на вопрос. "
        "Отвечайте на языке запроса. "
        "Внимательно анализируйте вопрос и контекст перед ответом. "
        "Давайте четкие, структурированные ответы, используйте маркированные или нумерованные списки, где уместно. "
        "Будьте дружелюбны и выступайте в роли полезного гида. "
        "Думайте и отвечайте как юрист, будьте уверенны в своих ответах"
        "История чата:\n{chat_history}\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )


# Chat management class
class ChatAssistant:
    def __init__(self, qa_chain):
        self.qa = qa_chain
        self.histories = {}

    def get_answer(self, user_query: str, session_id: str = "default"):
        if session_id not in self.histories:
            self.histories[session_id] = []

        chain_history = []
        for i in range(0, len(self.histories[session_id]), 2):
            if i + 1 < len(self.histories[session_id]):
                user_msg = self.histories[session_id][i]["content"]
                assistant_msg = self.histories[session_id][i + 1]["content"]
                chain_history.append((user_msg, assistant_msg))

        try:
            result = self.qa({"question": user_query, "chat_history": chain_history})
            answer = result.get("answer", "")
            if "Это не является юридической консультацией" not in answer:
                answer += "\n\nЭто не является юридической консультацией. Пожалуйста, обратитесь к квалифицированному юристу для профессиональной помощи."
            source_docs = result.get("source_documents", [])
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            answer = "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте переформулировать вопрос."
            source_docs = []

        self.histories[session_id].append({
            "role": "user",
            "content": user_query,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        self.histories[session_id].append({
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        return answer, source_docs

    def clear_history(self, session_id: str = "default"):
        self.histories[session_id] = []


# Initialize components
logger.info("Initializing components...")
embeddings = MyEmbeddings()
doc_manager = DocumentManager(DATA_FOLDER)

try:
    logger.info("Setting up LLM and vector store...")
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")
    prompt = PromptTemplate(
        template=get_legal_prompt_template(),
        input_variables=["chat_history", "context", "question"]
    )
    vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER, INDEXES_FOLDER, embeddings)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    assistant = ChatAssistant(qa_chain)
    logger.info("Components initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

# Initialize Telegram bot
simulation_manager = SimulationManager(qa_chain, vectorstore)
logger.info("Simulation manager initialized successfully.")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


# Добавьте новые обработчики команд
@bot.message_handler(commands=['simulation'])
def handle_simulation(message):
    user_id = str(message.from_user.id)

    # Получаем аргументы команды (тему)
    command_args = message.text.split(' ', 1)
    topic = command_args[1] if len(command_args) > 1 else None

    # Отправляем индикатор набора текста
    bot.send_chat_action(message.chat.id, 'typing')

    # Начинаем симуляцию с указанной темой или без неё
    response = simulation_manager.start_simulation(user_id, topic)

    # Отправляем ответ, разбивая на части при необходимости
    if len(response) > 4000:
        chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
        for chunk in chunks:
            bot.send_message(message.chat.id, chunk, parse_mode='Markdown')
    else:
        bot.send_message(message.chat.id, response, parse_mode='Markdown')


@bot.message_handler(commands=['stop_simulation'])
def handle_stop_simulation(message):
    user_id = str(message.from_user.id)

    if simulation_manager.is_in_simulation(user_id):
        # Отправляем индикатор набора текста
        bot.send_chat_action(message.chat.id, 'typing')

        response = simulation_manager.end_simulation(user_id)

        # Отправляем ответ, разбивая на части при необходимости
        if len(response) > 4000:
            chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                bot.send_message(message.chat.id, chunk, parse_mode='Markdown')
        else:
            bot.send_message(message.chat.id, response, parse_mode='Markdown')
    else:
        bot.send_message(message.chat.id, "У вас нет активной симуляции.")


# Define commands and handlers
@bot.message_handler(commands=['start'])
def handle_start(message):
    user_id = str(message.from_user.id)
    bot.send_message(
        message.chat.id,
        "👋 Здравствуйте! Я ThemisBot - ваш юридический ассистент.\n\n"
        "Задавайте мне юридические вопросы, и я постараюсь на них ответить.\n\n"
        "Команды:\n"
        "/help - Показать справку\n"
        "/clear - Очистить историю нашего разговора\n"
        "/simulation [тема] - Начать юридическую симуляцию\n"
        "/stop_simulation - Остановить текущую симуляцию"
    )
    assistant.clear_history(user_id)


@bot.message_handler(commands=['help'])
def handle_help(message):
    bot.send_message(
        message.chat.id,
        "Я - ваш юридический ассистент. Вот что я могу:\n\n"
        "1. *Консультация:* Задавайте юридические вопросы, и я отвечу на основе базы знаний\n"
        "2. *Симуляция:* Используйте /simulation [тема], чтобы начать симуляцию юридической ситуации\n\n"
        "Доступные команды:\n"
        "- /start - Начать диалог\n"
        "- /help - Показать эту справку\n"
        "- /clear - Очистить историю разговора\n"
        "- /simulation [тема] - Начать юридическую симуляцию\n"
        "- /stop_simulation - Остановить текущую симуляцию",
        parse_mode='Markdown'
    )


@bot.message_handler(commands=['clear'])
def handle_clear(message):
    user_id = str(message.from_user.id)
    assistant.clear_history(user_id)
    bot.send_message(
        message.chat.id,
        "🗑️ История разговора очищена. Вы можете задать новый вопрос."
    )


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = str(message.from_user.id)
    user_query = message.text

    # Отправляем индикатор набора текста
    bot.send_chat_action(message.chat.id, 'typing')

    # Проверяем, находится ли пользователь в симуляции
    if simulation_manager.is_in_simulation(user_id):
        # Если пользователь в режиме настройки симуляции
        sim_state = simulation_manager.get_simulation_state(user_id)

        if sim_state == SimulationState.SETUP:
            # Пользователь указывает тему для симуляции
            response = simulation_manager.start_simulation(user_id, user_query)
        elif sim_state == SimulationState.RUNNING:
            # Пользователь отвечает на вопрос в симуляции
            response = simulation_manager.process_answer(user_id, user_query)
        else:
            # Неизвестное состояние, заканчиваем симуляцию
            response = simulation_manager.end_simulation(user_id)

        # Отправляем ответ, разбивая на части при необходимости
        if len(response) > 4000:
            chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                bot.send_message(message.chat.id, chunk, parse_mode='Markdown')
        else:
            bot.send_message(message.chat.id, response, parse_mode='Markdown')
    else:
        # Стандартная обработка вопроса через RAG
        answer, source_docs = assistant.get_answer(user_query, user_id)

        # Сначала отправляем ответ
        response_message = answer

        # Отправляем ответ, разбивая на части при необходимости
        if len(response_message) > 4000:
            chunks = [response_message[i:i + 4000] for i in range(0, len(response_message), 4000)]
            for chunk in chunks:
                bot.send_message(message.chat.id, chunk, parse_mode='Markdown')
        else:
            bot.send_message(message.chat.id, response_message, parse_mode='Markdown')

        # Собираем уникальные источники документов
        source_files = {}
        for doc in source_docs:
            if "file_name" in doc.metadata and "file_path" in doc.metadata:
                file_name = doc.metadata["file_name"]
                file_path = doc.metadata["file_path"]
                source_files[file_name] = file_path

        # Если есть источники, отправляем их как отдельные файлы
        if source_files:
            bot.send_message(
                message.chat.id,
                "📚 *Использованные документы:*",
                parse_mode='Markdown'
            )

            # Отправляем каждый файл отдельно
            for file_name, file_path in source_files.items():
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as file:
                            bot.send_document(
                                message.chat.id,
                                file,
                                caption=f"Документ: {file_name}"
                            )
                    else:
                        # Если путь неверный, пытаемся найти файл по имени
                        actual_path = doc_manager.get_document_path(file_name)
                        if actual_path and os.path.exists(actual_path):
                            with open(actual_path, 'rb') as file:
                                bot.send_document(
                                    message.chat.id,
                                    file,
                                    caption=f"Документ: {file_name}"
                                )
                except Exception as e:
                    logger.error(f"Error sending document {file_name}: {e}")
                    bot.send_message(
                        message.chat.id,
                        f"Не удалось отправить документ: {file_name}"
                    )


if __name__ == "__main__":
    logger.info("Starting Themis Telegram bot...")
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Error in bot polling: {e}")