import gradio as gr
import os
import json
from datetime import datetime



def chat_with_bot(user_query, session_id="default"):
    answer = f"Это заглушка ответа на ваш вопрос: **{user_query}**"
    sources = "Документ 1, Документ 2"
    return answer, sources

def clear_chat(session_id="default"):
    pass


css = """
.gradio-container {
    background-color: #1a1a2e !important; 
    max-height: 100vh !important;
    overflow-y: auto;
}

/* Header */
h1, .badge, .header-subtitle {
    color: #e6e6e6 !important;
}

/* Chatbot panel */
#my-chat{
    color: white !important;
    background-color: #16213e !important;
    border: 1px solid #2a3b4d !important;
    border-radius: 8px !important;
    height: 400px !important;
    max-height: 400px !important;
    overflow-y: auto;
}
#my-chat.message {
    color: white !important;
}
/* Message styling */
#my-chat.message.user,
#my-chat.message.bot {
    color: white !important;
    background-color: #0f3460 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 16px !important;
    box-shadow: none !important;
}

/* Input area */
.input-box {
    background-color: #16213e !important;
    border: 1px solid #2a3b4d !important;
    border-radius: 8px !important;
}

/* Sidebar scroll */
.gr-column:last-child {
    max-height: 500px;
    overflow-y: auto;
}

/* Accordion and sidebar */
.sidebar, .gr-accordion {
    background-color: #16213e !important;
    border: 1px solid #2a3b4d !important;
    border-radius: 8px !important;
}

.gr-accordion .gr-accordion-header {
    background-color: #0f3460 !important;
    color: #ffffff !important;
    border-radius: 6px 6px 0 0 !important;
}

.gr-accordion .gr-accordion-panel {
    background-color: #16213e !important;
    color: #dddddd !important;
    border-radius: 0 0 6px 6px !important;
}

/* Buttons */
button {
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

button.primary {
    background-color: #4a6fa5 !important;
    color: white !important;
}

button.primary:hover {
    background-color: #3a5a8a !important;
}

button.secondary {
    background-color: #2a3b4d !important;
    color: white !important;
}

button.secondary:hover {
    background-color: #1a2a3d !important;
}

/* Inputs */
textarea, input[type="text"] {
    background-color: #0f3460 !important;
    color: white !important;
    border: 1px solid #2a3b4d !important;
    border-radius: 6px !important;
    padding: 12px !important;
}

/* Footer text */
.footer-text {
    color: #777777 !important;
    font-size: 0.8em !important;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="ThemisBot - Юридический Ассистент",
    css=css
) as demo:
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown(f"""
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                <img src="https://i.imgur.com/5L8TbWp.png" width="40"/>
                <div>
                    <h1 style="margin: 0;">ThemisBot</h1>
                    <span class="badge">Юридический ассистент</span>
                </div>
            </div>
            <p class="header-subtitle">
                Получайте быстрые юридические консультации на основе загруженных документов
            </p>
            """)
        with gr.Column(scale=2):
            gr.Markdown(f"""
            <div style="text-align: right; font-size: 0.8em; color: #888;">
                Версия 1.0<br>
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="my-chat",
                label="Чат с ThemisBot",
                height=400,
                show_copy_button=True,
                show_label=False,
                bubble_full_width=False,
                avatar_images=("🤖", "👤")
            )

            with gr.Row(equal_height=True):
                user_input = gr.Textbox(
                    placeholder="Введите ваш юридический вопрос...",
                    show_label=False,
                    lines=2,
                    max_lines=4,
                    container=False,
                    autofocus=True
                    
                )
                send_button = gr.Button("📨 Отправить", variant="primary", scale=0)

            with gr.Row():
                clear_btn = gr.Button("🧹 Очистить чат", variant="secondary")
                gr.Markdown("<div class='footer-text' style='text-align: right; flex-grow: 1;'>ThemisBot v1.0</div>")

        with gr.Column(scale=1):
            with gr.Accordion("📂 Доступные документы", open=True):
                gr.Markdown("Документы не загружены")

            with gr.Accordion("ℹ️ Как пользоваться", open=False):
                gr.Markdown("""
                **Советы для лучших ответов:**
                1. Формулируйте вопросы конкретно  
                2. Указывайте важные детали  
                3. Используйте юридические термины  

                **Примеры вопросов:**  
                - Как расторгнуть договор?  
                - Какие нужны документы?  
                - Какие есть основания?  
                """)

            with gr.Accordion("⚙️ Настройки", open=False):
                gr.Markdown("**Параметры системы:**")
                gr.Slider(minimum=1, maximum=5, value=3, label="Количество источников")
                gr.Checkbox(label="Показывать источники", value=True)

    def chat_response(user_query, chat_history):
        answer, sources = chat_with_bot(user_query, session_id="default")
        if sources:
            answer += f"\n\n🔍 Источники: {sources}"
        chat_history.append((user_query, answer))
        return chat_history, ""

    def clear_chat_history():
        clear_chat(session_id="default")
        return []

 
    send_button.click(
        fn=chat_response,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )

    user_input.submit(
        fn=chat_response,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )

    clear_btn.click(
        fn=clear_chat_history,
        inputs=None,
        outputs=[chatbot]
    )
    

if __name__ == "__main__":
    demo.launch()
