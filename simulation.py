import random
from typing import Dict, List, Any, Optional
from enum import Enum
import json
import re
import logging

# Настройка логирования
logger = logging.getLogger(__name__)


# Определение состояний симуляции
class SimulationState(Enum):
    INACTIVE = "inactive"  # Симуляция не активирована
    SETUP = "setup"  # Настройка симуляции
    RUNNING = "running"  # Симуляция активна
    FEEDBACK = "feedback"  # Предоставление обратной связи


class SimulationManager:
    def __init__(self, qa_chain, vectorstore):
        """
        Инициализирует менеджер симуляций.

        Args:
            qa_chain: Цепочка вопрос-ответ из LangChain
            vectorstore: Векторное хранилище с документами
        """
        self.qa_chain = qa_chain
        self.vectorstore = vectorstore
        self.simulation_sessions: Dict[str, Dict[str, Any]] = {}

    def start_simulation(self, user_id: str, topic: Optional[str] = None) -> str:
        """
        Начинает новую симуляцию для пользователя.

        Args:
            user_id: Идентификатор пользователя
            topic: Тема для симуляции (опционально)

        Returns:
            str: Сообщение для пользователя
        """
        # Если уже есть активная симуляция, очищаем ее
        if user_id in self.simulation_sessions:
            self.end_simulation(user_id)

        # Если тема не указана, запрашиваем у пользователя
        if not topic:
            self.simulation_sessions[user_id] = {
                "state": SimulationState.SETUP,
                "steps": [],
                "current_step": 0,
                "score": 0,
                "feedback": []
            }
            return ("Для начала юридической симуляции, укажите тему или проблему, "
                    "например: 'трудовое право', 'защита прав потребителей', 'жилищное право' и т.д.")

        # Создаем симуляцию на основе указанной темы
        return self._generate_simulation(user_id, topic)

    def _generate_simulation(self, user_id: str, topic: str) -> str:
        """
        Генерирует симуляцию на основе указанной темы и документов из базы знаний.

        Args:
            user_id: Идентификатор пользователя
            topic: Тема для симуляции

        Returns:
            str: Первый шаг симуляции или сообщение об ошибке
        """
        # Получаем релевантные документы из базы знаний
        docs = self.vectorstore.similarity_search(
            f"юридическая ситуация сценарий пример {topic}", k=3
        )

        context = "\n".join([doc.page_content for doc in docs])

        # Используем LLM для создания сценария симуляции
        prompt = f"""
        На основе следующего контекста, создайте реалистичную юридическую симуляцию по теме: {topic}.

        Контекст:
        {context}

        Создайте сценарий с 3-5 шагами, где пользователь должен принимать юридические решения.
        Для каждого шага укажите:
        1. Описание ситуации
        2. Вопрос к пользователю
        3. 2-3 варианта действий (если применимо)
        4. Правильный ответ с обоснованием

        Формат должен быть строго в JSON, используя только двойные кавычки:
        {{
          "steps": [
            {{
              "situation": "Описание ситуации",
              "question": "Вопрос пользователю",
              "options": ["Вариант 1", "Вариант 2", "Вариант 3"],
              "correct_answer": "Юридически верный ответ",
              "explanation": "Объяснение, почему этот ответ правильный"
            }},
            ... и т.д.
          ]
        }}

        Убедитесь, что JSON является валидным и содержит обязательные поля для каждого шага: situation, question, options, correct_answer, explanation.
        Верните ТОЛЬКО JSON без дополнительного текста.
        """

        try:
            result = self.qa_chain({"question": prompt, "chat_history": []})
            answer = result.get("answer", "")

            # Улучшенное извлечение JSON из ответа
            json_pattern = r'\{(?:[^{}]|"[^"]*"|\{(?:[^{}]|"[^"]*")*\})*\}'
            json_match = re.search(json_pattern, answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Заменяем одинарные кавычки на двойные, если они есть
                json_str = json_str.replace("'", '"')
            else:
                # Попытка найти что-то похожее на JSON
                logger.warning(f"Не удалось найти стандартный JSON в ответе: {answer}")
                # Используем более простой подход
                start_idx = answer.find('{')
                end_idx = answer.rfind('}')
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = answer[start_idx:end_idx + 1]
                    json_str = json_str.replace("'", '"')
                else:
                    raise ValueError("Не удалось найти JSON в ответе")

            # Очищаем JSON от возможных артефактов
            json_str = re.sub(r'```json\s*|\s*```', '', json_str)
            simulation_data = json.loads(json_str)

            # Проверяем, что данные имеют правильную структуру
            if not isinstance(simulation_data, dict) or "steps" not in simulation_data:
                raise ValueError("Неверный формат данных симуляции")

            # Проверка и установка значений по умолчанию для каждого шага
            for step in simulation_data["steps"]:
                if "situation" not in step:
                    step["situation"] = "Ситуация не указана"
                if "question" not in step:
                    step["question"] = "Вопрос не указан"
                if "options" not in step:
                    step["options"] = []
                if "correct_answer" not in step:
                    step["correct_answer"] = "Правильный ответ не указан"
                if "explanation" not in step:
                    step["explanation"] = "Объяснение не предоставлено"

            # Сохраняем симуляцию
            self.simulation_sessions[user_id] = {
                "state": SimulationState.RUNNING,
                "topic": topic,
                "steps": simulation_data["steps"],
                "current_step": 0,
                "score": 0,
                "feedback": [],
                "start_time": None  # Будет установлено при первом шаге
            }

            # Возвращаем первый шаг симуляции
            return self.get_current_step(user_id)

        except Exception as e:
            logger.error(f"Ошибка при создании симуляции: {str(e)}")
            return f"Не удалось создать симуляцию по теме '{topic}'. Пожалуйста, попробуйте другую тему или сформулируйте запрос иначе."

    def get_current_step(self, user_id: str) -> str:
        """
        Возвращает текущий шаг симуляции.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            str: Текст текущего шага или сообщение об ошибке
        """
        if user_id not in self.simulation_sessions:
            return "У вас нет активной симуляции. Используйте /simulation для начала."

        session = self.simulation_sessions[user_id]
        if session["state"] != SimulationState.RUNNING:
            return "Симуляция не запущена. Используйте /simulation для начала."

        current_step = session["current_step"]
        steps = session["steps"]

        if current_step >= len(steps):
            return self.end_simulation(user_id)

        step_data = steps[current_step]

        # Формируем сообщение с описанием ситуации и вопросом
        message = f"📝 *Юридическая симуляция: {session['topic']}*\n\n"
        message += f"*Шаг {current_step + 1}/{len(steps)}*\n\n"
        message += f"{step_data['situation']}\n\n"
        message += f"❓ *Вопрос:* {step_data['question']}\n\n"

        # Если есть варианты ответов, добавляем их
        if "options" in step_data and step_data["options"]:
            message += "*Варианты:*\n"
            for i, option in enumerate(step_data["options"], 1):
                message += f"{i}. {option}\n"

        message += "\n_Введите ваш ответ или решение. Если предложены варианты, можете указать номер варианта или дать развернутый ответ._"

        return message

    def process_answer(self, user_id: str, answer: str) -> str:
        """
        Обрабатывает ответ пользователя на текущий шаг симуляции.

        Args:
            user_id: Идентификатор пользователя
            answer: Ответ пользователя

        Returns:
            str: Обратная связь по ответу и следующий шаг
        """
        if user_id not in self.simulation_sessions:
            return "У вас нет активной симуляции. Используйте /simulation для начала."

        session = self.simulation_sessions[user_id]
        if session["state"] != SimulationState.RUNNING:
            return "Симуляция не запущена. Используйте /simulation для начала."

        current_step = session["current_step"]
        steps = session["steps"]

        if current_step >= len(steps):
            return self.end_simulation(user_id)

        step_data = steps[current_step]

        # Используем LLM для оценки ответа пользователя
        prompt = f"""
        В рамках юридической симуляции оцените ответ пользователя:

        Тема симуляции: {session["topic"]}

        Ситуация: {step_data["situation"]}

        Вопрос: {step_data["question"]}

        Правильный ответ: {step_data["correct_answer"]}

        Обоснование: {step_data.get("explanation", "")}

        Ответ пользователя: {answer}

        Оцените ответ пользователя по шкале от 0 до 10, где:
        - 0-3: Неверный, содержит юридические ошибки
        - 4-6: Частично верный, есть неточности
        - 7-10: Верный, юридически правильный

        Дайте подробную обратную связь с объяснением, что было правильно/неправильно и почему.
        Также дайте конкретные рекомендации по улучшению.

        Верните ТОЛЬКО JSON без дополнительного текста в формате:
        {{"score": 5, "feedback": "Текст обратной связи"}}

        Убедитесь, что JSON валидный и содержит только двойные кавычки.
        """

        try:
            result = self.qa_chain({"question": prompt, "chat_history": []})
            llm_response = result.get("answer", "")

            # Улучшенный поиск JSON в ответе
            json_pattern = r'\{(?:[^{}]|"[^"]*")*\}'
            json_match = re.search(json_pattern, llm_response, re.DOTALL)

            if not json_match:
                # Запасной вариант, если JSON не найден
                logger.warning(f"Не удалось найти JSON в ответе LLM: {llm_response}")
                score = 5  # Значение по умолчанию
                feedback = "К сожалению, не удалось автоматически оценить ваш ответ. " + \
                           f"Правильный ответ: {step_data['correct_answer']}"
            else:
                try:
                    json_str = json_match.group(0)
                    # Очистка от возможных специальных символов и экранирование кавычек
                    json_str = json_str.replace("'", '"')
                    feedback_data = json.loads(json_str)
                    score = int(feedback_data.get("score", 5))
                    feedback = feedback_data.get("feedback", "Оценка не может быть предоставлена.")
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка декодирования JSON: {str(e)}, JSON строка: {json_str}")
                    score = 5
                    feedback = "К сожалению, произошла ошибка при оценке вашего ответа. " + \
                               f"Правильный ответ: {step_data['correct_answer']}"

            # Сохраняем оценку и обратную связь
            session["score"] += score
            session["feedback"].append({
                "step": current_step,
                "question": step_data["question"],
                "user_answer": answer,
                "score": score,
                "feedback": feedback
            })

            # Формируем ответ пользователю
            message = f"✅ *Ваш ответ оценен на {score}/10 баллов*\n\n"
            message += f"*Обратная связь:*\n{feedback}\n\n"
            message += f"*Правильный ответ:*\n{step_data['correct_answer']}\n\n"

            # Переходим к следующему шагу
            session["current_step"] += 1

            # Если это был последний шаг, заканчиваем симуляцию
            if session["current_step"] >= len(steps):
                message += self.end_simulation(user_id)
            else:
                message += "\n" + self.get_current_step(user_id)

            return message

        except Exception as e:
            logger.error(f"Ошибка при оценке ответа: {str(e)}")
            return f"Произошла ошибка при оценке вашего ответа. Пожалуйста, попробуйте еще раз или используйте /stop_simulation для завершения симуляции."

    def end_simulation(self, user_id: str) -> str:
        """
        Завершает симуляцию и предоставляет итоговую оценку.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            str: Итоговая оценка и рекомендации
        """
        if user_id not in self.simulation_sessions:
            return "У вас нет активной симуляции."

        session = self.simulation_sessions[user_id]
        total_steps = len(session["steps"])

        if not total_steps:
            del self.simulation_sessions[user_id]
            return "Симуляция была отменена."

        # Вычисляем общий балл
        max_possible_score = total_steps * 10
        total_score = session["score"]
        percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        # Определяем общую оценку
        if percentage >= 90:
            overall_rating = "Отлично! Вы показали глубокое понимание юридических аспектов."
        elif percentage >= 75:
            overall_rating = "Хорошо! Вы продемонстрировали хорошее знание правовых норм."
        elif percentage >= 60:
            overall_rating = "Удовлетворительно. Есть некоторые пробелы в понимании юридических аспектов."
        else:
            overall_rating = "Требуется улучшение. Рекомендуем изучить дополнительные материалы по этой теме."

        # Формируем итоговое сообщение
        message = f"🏁 *Симуляция завершена: {session.get('topic', 'Юридическая симуляция')}*\n\n"
        message += f"*Итоговый балл:* {total_score}/{max_possible_score} ({percentage:.1f}%)\n\n"
        message += f"*Общая оценка:* {overall_rating}\n\n"

        # Добавляем детальную информацию по каждому шагу
        message += "*Детальная обратная связь:*\n\n"
        for i, feedback in enumerate(session.get("feedback", []), 1):
            message += f"*Шаг {i}:* {feedback.get('score', 0)}/10 - {feedback.get('question', 'Вопрос')}\n"

        # Добавляем рекомендации
        try:
            topic = session.get("topic", "юридические вопросы")
            prompt = f"""
            На основе результатов юридической симуляции по теме '{topic}' с общим баллом {percentage:.1f}%, 
            предоставьте 3-5 конкретных рекомендаций по улучшению юридических знаний в этой области. 
            Рекомендации должны быть краткими, но информативными, с указанием законодательных актов 
            или ресурсов, которые могут быть полезны для изучения.
            """

            result = self.qa_chain({"question": prompt, "chat_history": []})
            recommendations = result.get("answer", "")

            message += f"\n*Рекомендации:*\n{recommendations}"
        except Exception as e:
            logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
            message += "\n*Рекомендации:* Не удалось сформировать рекомендации."

        # Очищаем сессию
        del self.simulation_sessions[user_id]

        return message

    def is_in_simulation(self, user_id: str) -> bool:
        """
        Проверяет, находится ли пользователь в симуляции.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            bool: True, если пользователь в симуляции, иначе False
        """
        return user_id in self.simulation_sessions

    def get_simulation_state(self, user_id: str) -> Optional[SimulationState]:
        """
        Возвращает текущее состояние симуляции пользователя.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            SimulationState: Состояние симуляции или None, если симуляции нет
        """
        if user_id not in self.simulation_sessions:
            return None
        return self.simulation_sessions[user_id]["state"]