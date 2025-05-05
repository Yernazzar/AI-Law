import os
from typing import List, Dict, Optional
from pathlib import Path

# Путь к папке с шаблонами
TEMPLATES_DIR = "templates"

# Словарь с маппингом ключевых слов запроса на шаблоны
TEMPLATE_MAPPING = {
    "развод": [
        {"name": "Иск о расторжении брака", "file": "Иск о расторжении брака.docx"},
        {"name": "заявление на расторжение брака", "file": "расторжение брака.pdf"},
        {"name": "Заявление о разделе совместно нажитого имущества", "file": "Заявление о разделе совместно нажитого имущества.docx"}
    ],
    "брачный договор": [
        {"name": "Брачный договор с разделным режимом имущества супругов", "file": "Брачный договор с разделным режимом имущества супругов.docx"}
    ],
    "доверенность": [
        {"name": "Доверенность на представление интересов", "file": "Доверенность на представление интересов.docx"},
    ],
    "родительские права": [
        {"name": "Иск о лишении родительских прав", "file": "Иск о лишении родительских прав.docx"}
    ],
    "алименты": [
        {"name": "Взыскание алиментов", "file": "взыскание алиментов.docx"},
        {"name": "Заявление о выдаче судебного приказа на алименты", "file": "Заявление о выдаче судебного приказа на алименты.doc"}
    ],
    "наследство":[  
        {"name": "Образец заявления об установлении юридического факта принятия наследства", "file": "Образец заявления об установлении юридического факта принятия наследства.docx"},
         {"name": "Заявление об установлении факта принятия наследства", "file": "Заявл. об устан. факта прин. наследства.docx"},
        {"name": "Заявление о выдаче судебного приказа на алименты", "file": "Заявление о выдаче судебного приказа на алименты.doc"}
        

    ]
    
}

def get_templates_dir() -> Path:
    """Возвращает путь к папке с шаблонами."""
    return Path(TEMPLATES_DIR)

def list_available_templates() -> List[Dict[str, str]]:
    """Возвращает список всех доступных шаблонов."""
    templates = []
    for files in TEMPLATE_MAPPING.values():
        templates.extend(files)
    return templates

def search_templates(query: str) -> List[Dict[str, str]]:
    """Ищет шаблоны по запросу пользователя."""
    query = query.lower().strip()
    matched_templates = []
    
    for keyword, templates in TEMPLATE_MAPPING.items():
        if keyword in query:
            matched_templates.extend(templates)
    
    # Удаление дубликатов, если шаблон соответствует нескольким ключевым словам
    unique_templates = {tpl["file"]: tpl for tpl in matched_templates}
    return list(unique_templates.values())

def get_template_path(template_file: str) -> Optional[Path]:
    """Возвращает полный путь к файлу шаблона, если он существует."""
    template_path = get_templates_dir() / template_file
    if template_path.exists():
        return template_path
    return None

def generate_template_response(query: str) -> Dict[str, any]:
    """
    Основная функция для обработки запроса и генерации ответа с шаблонами.
    Возвращает словарь с списком шаблонов и их путями.
    """
    templates = search_templates(query)
    
    if not templates:
        return {
            "success": False,
            "message": "Подходящие шаблоны не найдены. Уточните запрос.",
            "templates": []
        }
    
    response = {
        "success": True,
        "message": "Найдены следующие шаблоны документов:",
        "templates": []
    }
    
    for template in templates:
        template_path = get_template_path(template["file"])
        if template_path:
            response["templates"].append({
                "name": template["name"],
                "file": str(template_path)
            })
        else:
            response["templates"].append({
                "name": template["name"],
                "file": None,
                "error": f"Файл {template['file']} не найден в папке {TEMPLATES_DIR}"
            })
    
    return response

def validate_templates_dir() -> bool:
    """Проверяет существование папки с шаблонами."""
    return get_templates_dir().exists()