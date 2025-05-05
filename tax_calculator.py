from decimal import Decimal, ROUND_HALF_UP
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Константы (Казахстан, 2025)
MRP = Decimal("3932")  # Месячный расчетный показатель
MZP = Decimal("85000")  # Минимальная зарплата
IPN_RATE = Decimal("0.10")  # Базовая ставка ИПН
IPN_HIGH_RATE = Decimal("0.15")  # Ставка для доходов > 34 млн тенге
IPN_HIGH_THRESHOLD = Decimal("34000000")  # Порог для 15% ИПН
OPV_RATE = Decimal("0.10")  # Ставка ОПВ
OPV_MAX = 50 * MRP  # Максимум ОПВ (50 МРП)
SO_RATE = Decimal("0.035")  # Ставка СО
SO_MAX = 7 * MRP  # Максимум СО (7 МРП)
OSMS_RATE = Decimal("0.03")  # Ставка ОСМС
OSMS_MAX = 10 * MRP  # Максимум ОСМС
IPN_DEDUCTION = 14 * MRP  # Стандартный вычет ИПН
IPN_DISABILITY_DEDUCTION = 882 * MRP  # Вычет для инвалидов
UPROSCHENKA_RATE = Decimal("0.03")  # Ставка для ИП на упрощенке (1.5% ИПН + 1.5% соцналог)
OSMS_IP_RATE = Decimal("0.05")  # Ставка ОСМС для ИП за себя
OSMS_IP_BASE = Decimal("1.4") * MZP  # База для ОСМС ИП

def validate_input(value: str, input_type: str) -> tuple[bool, str, any]:
    """
    Проверяет корректность ввода и возвращает преобразованное значение.
    Args:
        value: Введенное значение
        input_type: 'decimal' или 'int'
    Returns:
        Tuple: (успех, сообщение об ошибке или '', преобразованное значение или None)
    """
    try:
        cleaned_value = value.replace(",", ".").replace(" ", "").strip()
        if input_type == "decimal":
            result = Decimal(cleaned_value)
            if result < 0:
                return False, "Введите неотрицательное число.", None
            return True, "", result
        elif input_type == "int":
            result = int(float(cleaned_value))
            if result < 0:
                return False, "Введите неотрицательное целое число.", None
            return True, "", Decimal(result)
    except (ValueError, TypeError):
        return False, f"Введите {'число' if input_type == 'decimal' else 'целое число'} (например, {'100000.50' if input_type == 'decimal' else '1'}).", None

def calculate_ipn(income: Decimal, is_resident: bool = True, has_deduction: bool = True, is_disabled: bool = False) -> tuple[Decimal, str]:
    """
    Рассчитывает ИПН для физлица.
    Args:
        income: Годовой доход (тенге)
        is_resident: Резидент РК (для вычета 14 МРП)
        has_deduction: Применяется ли вычет 14 МРП
        is_disabled: Инвалид (для вычета 882 МРП)
    Returns:
        Tuple: (сумма ИПН, описание расчета)
    """
    if income < 0:
        return Decimal("0"), "Ошибка: доход не может быть отрицательным."

    deductions = Decimal("0")
    description = []

    if is_resident and has_deduction:
        deductions += IPN_DEDUCTION
        description.append(f"Стандартный вычет: {IPN_DEDUCTION:,.0f} тенге")
    if is_disabled:
        deductions += IPN_DISABILITY_DEDUCTION
        description.append(f"Вычет для инвалидов: {IPN_DISABILITY_DEDUCTION:,.0f} тенге")

    taxable_income = max(Decimal("0"), income - deductions)
    tax = Decimal("0")

    if taxable_income <= 0:
        return tax, "ИПН: 0 тенге (доход меньше вычетов)"

    if taxable_income <= IPN_HIGH_THRESHOLD:
        tax = taxable_income * IPN_RATE
        description.append(f"ИПН: {taxable_income:,.2f} × {IPN_RATE*100}% = {tax:,.2f} тенге")
    else:
        base_tax = IPN_HIGH_THRESHOLD * IPN_RATE
        high_tax = (taxable_income - IPN_HIGH_THRESHOLD) * IPN_HIGH_RATE
        tax = base_tax + high_tax
        description.append(
            f"ИПН (до {IPN_HIGH_THRESHOLD:,.0f} тенге): {IPN_HIGH_THRESHOLD:,.0f} × {IPN_RATE*100}% = {base_tax:,.2f} тенге"
        )
        description.append(
            f"ИПН (свыше): {(taxable_income - IPN_HIGH_THRESHOLD):,.2f} × {IPN_HIGH_RATE*100}% = {high_tax:,.2f} тенге"
        )
        description.append(f"Итого ИПН: {tax:,.2f} тенге")

    return tax.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "\n".join(description)

def calculate_opv(income: Decimal) -> tuple[Decimal, str]:
    """
    Рассчитывает ОПВ.
    Args:
        income: Месячный доход (тенге)
    Returns:
        Tuple: (сумма ОПВ, описание расчета)
    """
    if income < 0:
        return Decimal("0"), "Ошибка: доход не может быть отрицательным."

    opv = min(income * OPV_RATE, OPV_MAX)
    description = f"ОПВ: {income:,.2f} × {OPV_RATE*100}% = {opv:,.2f} тенге (макс. {OPV_MAX:,.0f} тенге)"

    return opv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), description

def calculate_so(income: Decimal, is_uproschenka: bool = False) -> tuple[Decimal, str]:
    """
    Рассчитывает СО.
    Args:
        income: Месячный доход (тенге)
        is_uproschenka: ИП на упрощенке (СО не платится)
    Returns:
        Tuple: (сумма СО, описание расчета)
    """
    if income < 0:
        return Decimal("0"), "Ошибка: доход не может быть отрицательным."

    if is_uproschenka:
        return Decimal("0"), "СО: 0 тенге (не платится для ИП на упрощенке)"

    so = min(income * SO_RATE, SO_MAX)
    description = f"СО: {income:,.2f} × {SO_RATE*100}% = {so:,.2f} тенге (макс. {SO_MAX:,.0f} тенге)"

    return so.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), description

def calculate_osms(income: Decimal) -> tuple[Decimal, str]:
    """
    Рассчитывает ОСМС.
    Args:
        income: Месячный доход (тенге)
    Returns:
        Tuple: (сумма ОСМС, описание расчета)
    """
    if income < 0:
        return Decimal("0"), "Ошибка: доход не может быть отрицательным."

    osms = min(income * OSMS_RATE, OSMS_MAX)
    description = f"ОСМС: {income:,.2f} × {OSMS_RATE*100}% = {osms:,.2f} тенге (макс. {OSMS_MAX:,.0f} тенге)"

    return osms.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), description

def calculate_ip_uproschenka(income: Decimal) -> tuple[Decimal, str]:
    """
    Рассчитывает налоги для ИП на упрощенке.
    Args:
        income: Месячный доход (тенге)
    Returns:
        Tuple: (сумма налога, описание расчета)
    """
    if income < 0:
        return Decimal("0"), "Ошибка: доход не может быть отрицательным."

    tax = income * UPROSCHENKA_RATE
    osms = OSMS_IP_RATE * OSMS_IP_BASE
    total = tax + osms
    description = [
        f"Налог (ИПН+соцналог): {income:,.2f} × {UPROSCHENKA_RATE*100}% = {tax:,.2f} тенге",
        f"ОСМС за себя: {OSMS_IP_BASE:,.2f} × {OSMS_IP_RATE*100}% = {osms:,.2f} тенге",
        f"Итого: {total:,.2f} тенге"
    ]

    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "\n".join(description)

def calculate_salary_net(gross: Decimal, is_resident: bool = True, has_deduction: bool = True, is_disabled: bool = False, is_uproschenka: bool = False) -> tuple[Decimal, str]:
    """
    Рассчитывает зарплату "на руки" (прямой метод).
    Args:
        gross: Месячный оклад до вычетов (тенге)
        is_resident: Резидент РК
        has_deduction: Применяется ли вычет 14 МРП
        is_disabled: Инвалид
        is_uproschenka: ИП на упрощенке
    Returns:
        Tuple: (чистая зарплата, описание расчета)
    """
    if gross < 0:
        return Decimal("0"), "Ошибка: оклад не может быть отрицательным."

    # ОПВ
    opv, opv_desc = calculate_opv(gross)
    income_after_opv = gross - opv

    # СО
    so, so_desc = calculate_so(income_after_opv, is_uproschenka)
    income_after_so = income_after_opv - so

    # ОСМС
    osms, osms_desc = calculate_osms(income_after_opv)
    income_after_osms = income_after_so - osms

    # ИПН (годовой доход для прогрессивной шкалы)
    annual_income = income_after_opv * 12
    ipn, ipn_desc = calculate_ipn(annual_income, is_resident, has_deduction, is_disabled)
    monthly_ipn = ipn / 12

    net_salary = income_after_osms - monthly_ipn
    description = [
        f"Оклад: {gross:,.2f} тенге",
        opv_desc,
        so_desc,
        osms_desc,
        f"ИПН (месячный): {ipn_desc.replace('ИПН', 'ИПН (годовой)')}\nМесячный ИПН: {monthly_ipn:,.2f} тенге",
        f"Зарплата на руки: {net_salary:,.2f} тенге"
    ]

    return net_salary.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "\n".join(description)

def calculate_salary_gross(net: Decimal, is_resident: bool = True, has_deduction: bool = True, is_disabled: bool = False, is_uproschenka: bool = False) -> tuple[Decimal, str]:
    """
    Рассчитывает оклад до вычетов (обратный метод).
    Args:
        net: Желаемая зарплата "на руки" (тенге)
        is_resident: Резидент РК
        has_deduction: Применяется ли вычет 14 МРП
        is_disabled: Инвалид
        is_uproschenka: ИП на упрощенке
    Returns:
        Tuple: (оклад, описание расчета)
    """
    if net < 0:
        return Decimal("0"), "Ошибка: сумма на руки не может быть отрицательной."

    # Итеративный поиск оклада
    def estimate_gross(guess: Decimal) -> Decimal:
        net_salary, _ = calculate_salary_net(guess, is_resident, has_deduction, is_disabled, is_uproschenka)
        return net_salary - net

    # Бинарный поиск
    low, high = net, net * Decimal("2")
    max_iterations = 100
    tolerance = Decimal("0.01")

    for _ in range(max_iterations):
        mid = (low + high) / 2
        diff = estimate_gross(mid)
        if abs(diff) < tolerance:
            gross = mid
            break
        if diff > 0:
            high = mid
        else:
            low = mid
    else:
        return Decimal("0"), "Ошибка: не удалось рассчитать оклад."

    # Финальный расчет
    net_salary, description = calculate_salary_net(gross, is_resident, has_deduction, is_disabled, is_uproschenka)
    description = f"Расчет оклада для {net:,.2f} тенге на руки:\n{description}"

    return gross.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), description

def calculate_tax(tax_type: str, params: dict) -> tuple[Decimal, str]:
    """
    Универсальная функция для расчета налога или зарплаты.
    Args:
        tax_type: 'ipn', 'opv', 'so', 'osms', 'ip_uproschenka', 'salary_net', 'salary_gross'
        params: Словарь с параметрами
    Returns:
        Tuple: (сумма, описание расчета)
    """
    try:
        if tax_type == "ipn":
            income = params.get("income", Decimal("0"))
            is_resident = params.get("is_resident", True)
            has_deduction = params.get("has_deduction", True)
            is_disabled = params.get("is_disabled", False)
            return calculate_ipn(income, is_resident, has_deduction, is_disabled)
        elif tax_type == "opv":
            income = params.get("income", Decimal("0"))
            return calculate_opv(income)
        elif tax_type == "so":
            income = params.get("income", Decimal("0"))
            is_uproschenka = params.get("is_uproschenka", False)
            return calculate_so(income, is_uproschenka)
        elif tax_type == "osms":
            income = params.get("income", Decimal("0"))
            return calculate_osms(income)
        elif tax_type == "ip_uproschenka":
            income = params.get("income", Decimal("0"))
            return calculate_ip_uproschenka(income)
        elif tax_type == "salary_net":
            gross = params.get("gross", Decimal("0"))
            is_resident = params.get("is_resident", True)
            has_deduction = params.get("has_deduction", True)
            is_disabled = params.get("is_disabled", False)
            is_uproschenka = params.get("is_uproschenka", False)
            return calculate_salary_net(gross, is_resident, has_deduction, is_disabled, is_uproschenka)
        elif tax_type == "salary_gross":
            net = params.get("net", Decimal("0"))
            is_resident = params.get("is_resident", True)
            has_deduction = params.get("has_deduction", True)
            is_disabled = params.get("is_disabled", False)
            is_uproschenka = params.get("is_uproschenka", False)
            return calculate_salary_gross(net, is_resident, has_deduction, is_disabled, is_uproschenka)
        else:
            return Decimal("0"), f"Ошибка: неизвестный тип расчета '{tax_type}'."
    except Exception as e:
        logging.error(f"Error calculating {tax_type}: {str(e)}")
        return Decimal("0"), f"Ошибка при расчете: {str(e)}"