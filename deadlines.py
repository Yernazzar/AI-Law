import datetime
from typing import List, Dict, Tuple, Optional
import json
import os

# Define the structure of a deadline
class Deadline:
    def __init__(self, name: str, description: str, calculation_func, category: str):
        """
        Initialize a deadline object.

        Args:
            name: Short name of the deadline
            description: Detailed description of the deadline
            calculation_func: Function that calculates the deadline date based on current date
            category: Category of the deadline (e.g., "Tax", "Business", "Legal")
        """
        self.name = name
        self.description = description
        self.calculation_func = calculation_func
        self.category = category

    def get_date_and_remaining(self) -> Tuple[datetime.date, int]:
        """
        Calculate the deadline date and days remaining.

        Returns:
            Tuple of (deadline_date, days_remaining)
        """
        deadline_date = self.calculation_func()
        today = datetime.date.today()
        days_remaining = (deadline_date - today).days
        return deadline_date, days_remaining

    def format_remaining_time(self) -> str:
        """
        Format the remaining time in a human-readable format.

        Returns:
            String describing remaining time (e.g., "3 месяца и 2 дня")
        """
        _, days_remaining = self.get_date_and_remaining()

        if days_remaining < 0:
            return "Срок уже истек"

        years = days_remaining // 365
        months = (days_remaining % 365) // 30
        days = (days_remaining % 365) % 30

        time_parts = []
        if years > 0:
            years_str = f"{years} " + self._pluralize(years, "год", "года", "лет")
            time_parts.append(years_str)

        if months > 0:
            months_str = f"{months} " + self._pluralize(months, "месяц", "месяца", "месяцев")
            time_parts.append(months_str)

        if days > 0 or not time_parts:
            days_str = f"{days} " + self._pluralize(days, "день", "дня", "дней")
            time_parts.append(days_str)

        if len(time_parts) == 1:
            return time_parts[0]
        else:
            return " и ".join([" ".join(time_parts[:-1]), time_parts[-1]])

    def _pluralize(self, n: int, form1: str, form2: str, form5: str) -> str:
        """
        Return the correct plural form for Russian language.

        Args:
            n: Number
            form1: Form for 1
            form2: Form for 2-4
            form5: Form for 5+

        Returns:
            Correct form
        """
        n = abs(n) % 100
        if 11 <= n <= 19:
            return form5
        n = n % 10
        if n == 1:
            return form1
        if 2 <= n <= 4:
            return form2
        return form5


# Helper functions to calculate various deadlines
def _next_quarter_end() -> datetime.date:
    """Calculate the end of the current quarter plus 15 days."""
    today = datetime.date.today()
    year = today.year
    quarter = (today.month - 1) // 3 + 1
    next_quarter = quarter + 1
    next_quarter_year = year

    if next_quarter > 4:
        next_quarter = 1
        next_quarter_year += 1

    month = 3 * next_quarter - 2
    quarter_start = datetime.date(next_quarter_year, month, 1)
    return quarter_start + datetime.timedelta(days=14)  # 15th day of the first month of the next quarter


def _next_annual_tax_declaration() -> datetime.date:
    """Calculate the deadline for annual tax declaration (March 31)."""
    today = datetime.date.today()
    year = today.year
    deadline = datetime.date(year, 3, 31)

    if today > deadline:
        # If today is past March 31, the next deadline is next year
        return datetime.date(year + 1, 3, 31)
    return deadline


def _company_annual_reporting() -> datetime.date:
    """Calculate the deadline for company annual reporting (April 30)."""
    today = datetime.date.today()
    year = today.year
    deadline = datetime.date(year, 4, 30)

    if today > deadline:
        # If today is past April 30, the next deadline is next year
        return datetime.date(year + 1, 4, 30)
    return deadline


def _next_property_tax() -> datetime.date:
    """Calculate the deadline for property tax (October 1)."""
    today = datetime.date.today()
    year = today.year
    deadline = datetime.date(year, 10, 1)

    if today > deadline:
        # If today is past October 1, the next deadline is next year
        return datetime.date(year + 1, 10, 1)
    return deadline


def _next_personal_income_tax() -> datetime.date:
    """Calculate the deadline for personal income tax (July 31)."""
    today = datetime.date.today()
    year = today.year
    deadline = datetime.date(year, 7, 31)

    if today > deadline:
        # If today is past July 31, the next deadline is next year
        return datetime.date(year + 1, 7, 31)
    return deadline


def _ip_registration_update() -> datetime.date:
    """Calculate the annual deadline for updating IP registration data (January 31)."""
    today = datetime.date.today()
    year = today.year
    deadline = datetime.date(year, 1, 31)

    if today > deadline:
        # If today is past January 31, the next deadline is next year
        return datetime.date(year + 1, 1, 31)
    return deadline


def _end_of_month() -> datetime.date:
    """Calculate the end of the current month."""
    today = datetime.date.today()
    next_month = today.replace(day=28) + datetime.timedelta(days=4)  # Move to next month
    return next_month - datetime.timedelta(days=next_month.day)  # Last day of current month


def _vat_declaration() -> datetime.date:
    """Calculate the deadline for VAT declaration (15th of the second month after the reporting quarter)."""
    today = datetime.date.today()
    year = today.year
    quarter = (today.month - 1) // 3 + 1

    if quarter == 1:
        deadline_month = 5  # May for Q1
    elif quarter == 2:
        deadline_month = 8  # August for Q2
    elif quarter == 3:
        deadline_month = 11  # November for Q3
    else:  # quarter == 4
        deadline_month = 2  # February next year for Q4
        if deadline_month < today.month:
            year += 1

    return datetime.date(year, deadline_month, 15)


# Define all available deadlines
DEADLINES = [
    Deadline(
        name="Квартальная налоговая отчетность",
        description="Сдача квартальной налоговой отчетности для юридических лиц и ИП на ОСН",
        calculation_func=_next_quarter_end,
        category="Налоги"
    ),
    Deadline(
        name="Годовая налоговая декларация",
        description="Срок подачи годовой налоговой декларации для физических лиц",
        calculation_func=_next_annual_tax_declaration,
        category="Налоги"
    ),
    Deadline(
        name="Годовая финансовая отчетность",
        description="Срок сдачи годовой финансовой отчетности для юридических лиц",
        calculation_func=_company_annual_reporting,
        category="Бизнес"
    ),
    Deadline(
        name="Налог на имущество",
        description="Срок уплаты налога на имущество физических лиц",
        calculation_func=_next_property_tax,
        category="Налоги"
    ),
    Deadline(
        name="Подоходный налог",
        description="Срок уплаты индивидуального подоходного налога для физических лиц",
        calculation_func=_next_personal_income_tax,
        category="Налоги"
    ),
    Deadline(
        name="Обновление данных ИП",
        description="Обязательное ежегодное обновление регистрационных данных ИП",
        calculation_func=_ip_registration_update,
        category="Бизнес"
    ),
    Deadline(
        name="Ежемесячная отчетность",
        description="Срок подачи ежемесячной отчетности для работодателей по ОСМС, ОПВС и социальному налогу",
        calculation_func=_end_of_month,
        category="Налоги"
    ),
    Deadline(
        name="Декларация по НДС",
        description="Срок подачи декларации по НДС для налогоплательщиков, являющихся плательщиками НДС",
        calculation_func=_vat_declaration,
        category="Налоги"
    )
]


def get_all_deadlines() -> List[Dict]:
    """
    Get all deadlines with their dates and remaining time.

    Returns:
        List of dictionaries with deadline information
    """
    result = []
    for deadline in DEADLINES:
        date, days_remaining = deadline.get_date_and_remaining()
        result.append({
            "name": deadline.name,
            "description": deadline.description,
            "category": deadline.category,
            "date": date.strftime("%d.%m.%Y"),
            "days_remaining": days_remaining,
            "formatted_time": deadline.format_remaining_time()
        })

    # Sort by days remaining (closest first)
    result.sort(key=lambda x: x["days_remaining"])
    return result


def get_deadlines_by_category(category: str) -> List[Dict]:
    """
    Get deadlines filtered by category.

    Args:
        category: Category to filter by

    Returns:
        List of dictionaries with deadline information for the specified category
    """
    return [d for d in get_all_deadlines() if d["category"].lower() == category.lower()]


def generate_deadlines_message() -> str:
    """
    Generate a formatted message with all deadlines.

    Returns:
        String with formatted deadlines information
    """
    deadlines = get_all_deadlines()

    message = "📅 *Важные юридические сроки:*\n\n"

    for deadline in deadlines:
        message += f"*{deadline['name']}*\n"
        message += f"🗓️ Дата: {deadline['date']}\n"
        message += f"⏱️ Осталось: {deadline['formatted_time']}\n"
        message += f"📋 {deadline['description']}\n\n"

    message += "_Обратите внимание: информация о сроках является приблизительной. Для точных дат обратитесь к официальным источникам._"

    return message


def generate_category_deadlines_message(category: str) -> str:
    """
    Generate a formatted message with deadlines for a specific category.

    Args:
        category: Category to filter by

    Returns:
        String with formatted deadlines information for the specified category
    """
    deadlines = get_deadlines_by_category(category)

    if not deadlines:
        return f"Сроки для категории '{category}' не найдены."

    message = f"📅 *Важные сроки в категории '{category}':*\n\n"

    for deadline in deadlines:
        message += f"*{deadline['name']}*\n"
        message += f"🗓️ Дата: {deadline['date']}\n"
        message += f"⏱️ Осталось: {deadline['formatted_time']}\n"
        message += f"📋 {deadline['description']}\n\n"

    message += "_Обратите внимание: информация о сроках является приблизительной. Для точных дат обратитесь к официальным источникам._"

    return message


def generate_pdf_report():
    """
    Generate a PDF report with all deadlines.

    Returns:
        BytesIO object containing the PDF file
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from io import BytesIO
    import os

    # Try to use a Cyrillic-compatible font if available
    try:
        if os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
            pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
            font_name = 'DejaVu'
        else:
            font_name = 'Helvetica'
    except:
        font_name = 'Helvetica'

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont(font_name, 16)
    c.drawString(2 * cm, height - 2 * cm, "Юридические сроки и дедлайны")

    # Date of generation
    c.setFont(font_name, 10)
    c.drawString(2 * cm, height - 3 * cm, f"Дата формирования: {datetime.date.today().strftime('%d.%m.%Y')}")

    # Deadlines
    deadlines = get_all_deadlines()
    y = height - 4 * cm

    for deadline in deadlines:
        if y < 4 * cm:  # Check if we need a new page
            c.showPage()
            c.setFont(font_name, 16)
            c.drawString(2 * cm, height - 2 * cm, "Юридические сроки и дедлайны (продолжение)")
            y = height - 4 * cm

        c.setFont(font_name, 12)
        c.drawString(2 * cm, y, deadline['name'])
        y -= 0.8 * cm

        c.setFont(font_name, 10)
        c.drawString(2 * cm, y, f"Дата: {deadline['date']}")
        y -= 0.6 * cm

        c.drawString(2 * cm, y, f"Осталось: {deadline['formatted_time']}")
        y -= 0.6 * cm

        # Split description into multiple lines if necessary
        description = deadline['description']
        max_width = width - 4 * cm
        lines = []
        current_line = ""

        for word in description.split():
            test_line = current_line + " " + word if current_line else word
            if c.stringWidth(test_line, font_name, 10) < max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        for line in lines:
            c.drawString(2 * cm, y, line)
            y -= 0.6 * cm

        c.drawString(2 * cm, y, f"Категория: {deadline['category']}")
        y -= 1.2 * cm  # Extra space between deadlines

    # Footer
    c.setFont(font_name, 8)
    c.drawString(2 * cm, 2 * cm, "Обратите внимание: информация о сроках является приблизительной.")
    c.drawString(2 * cm, 1.7 * cm, "Для точных дат обратитесь к официальным источникам.")

    c.save()
    buffer.seek(0)
    return buffer


def get_categories() -> List[str]:
    """
    Get all available deadline categories.

    Returns:
        List of unique category names
    """
    return sorted(list(set(deadline.category for deadline in DEADLINES)))


USER_DEADLINES_FILE = "user_deadlines.json"


def load_user_deadlines():
    """Загрузка пользовательских дедлайнов из файла."""
    if not os.path.exists(USER_DEADLINES_FILE):
        return {}

    try:
        with open(USER_DEADLINES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке пользовательских дедлайнов: {e}")
        return {}


def save_user_deadlines(user_deadlines):
    """Сохранение пользовательских дедлайнов в файл."""
    try:
        with open(USER_DEADLINES_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_deadlines, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении пользовательских дедлайнов: {e}")
        return False


def add_user_deadline(user_id, title, date, category="Пользовательский"):
    """Добавление нового пользовательского дедлайна."""
    user_deadlines = load_user_deadlines()
    user_id = str(user_id)  # Преобразуем в строку для использования в качестве ключа

    if user_id not in user_deadlines:
        user_deadlines[user_id] = []

    # Добавляем новый дедлайн
    user_deadlines[user_id].append({
        "title": title,
        "date": date,
        "category": category,
        "created_at": datetime.now().isoformat()
    })

    return save_user_deadlines(user_deadlines)


def delete_user_deadline(user_id, deadline_index):
    """Удаление пользовательского дедлайна по индексу."""
    user_deadlines = load_user_deadlines()
    user_id = str(user_id)

    if user_id not in user_deadlines or deadline_index >= len(user_deadlines[user_id]):
        return False

    user_deadlines[user_id].pop(deadline_index)
    return save_user_deadlines(user_deadlines)


def get_user_deadlines(user_id):
    """Получение списка пользовательских дедлайнов."""
    user_deadlines = load_user_deadlines()
    return user_deadlines.get(str(user_id), [])


def generate_user_deadlines_message(user_id):
    """Генерация сообщения со списком пользовательских дедлайнов."""
    deadlines = get_user_deadlines(user_id)

    if not deadlines:
        return "У вас нет сохраненных дедлайнов. Чтобы добавить новый, используйте команду /add_deadline"

    message = "📋 *Ваши персональные дедлайны:*\n\n"

    for i, deadline in enumerate(deadlines):
        message += f"{i + 1}. *{deadline['title']}*\n"
        message += f"   Дата: {deadline['date']}\n"
        message += f"   Категория: {deadline['category']}\n\n"

    return message