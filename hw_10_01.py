import pulp

# Визначення моделі
model = pulp.LpProblem("Максимізація виробництва напоїв", pulp.LpMaximize)

# Змінні для кількості виробленого Лимонаду та Фруктового соку
lemonade = pulp.LpVariable("Лимонад", lowBound=0, cat='Integer')
fruit_juice = pulp.LpVariable("Фруктовий сік", lowBound=0, cat='Integer')

# Функція цілі: максимізувати загальну кількість вироблених напоїв
model += lemonade + fruit_juice, "Загальна кількість продуктів"

# Обмеження на ресурси
model += 2 * lemonade + 1 * fruit_juice <= 100, "Обмеження води"
model += 1 * lemonade <= 50, "Обмеження цукру"
model += 1 * lemonade <= 30, "Обмеження лимонного соку"
model += 2 * fruit_juice <= 40, "Обмеження фруктового пюре"

# Розв'язання задачі
model.solve()

# Отримання та виведення результатів
lemonade_quantity = pulp.value(lemonade)
fruit_juice_quantity = pulp.value(fruit_juice)
total_products = pulp.value(model.objective)

print(f"Виробництво лимонаду: {lemonade_quantity}")
print(f"Виробництво фруктового соку: {fruit_juice_quantity}")
print(f"Загальна кількість продуктів: {total_products}")