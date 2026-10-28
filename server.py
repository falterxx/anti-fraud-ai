from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # Разрешаем запросы из браузера

# Обучаем модель при запуске сервера
print("Загрузка данных и обучение антифрод-модели...")
df = pd.read_csv('creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Модель ИИ успешно обучена и готова к работе!")

# Функция извлечения времени и суммы из текста сообщения
def parse_transaction(text):
    lower = text.lower()
    amount = 5000.0
    time = 12

    # Ищем сумму (например, 485000 тенге / ₸)
    amount_match = re.search(r'(\d+[\s.,]?\d*)\s*(тенге|тг|tg|руб|тыс|k|₸)?', lower)
    if amount_match:
        try:
            num = float(amount_match.group(1).replace(' ', '').replace(',', '.'))
            if 'тыс' in lower or 'k' in lower:
                num *= 1000
            amount = num
        except ValueError:
            pass

    # Ищем время
    time_match = re.search(r'(\d{1,2})\s*(час|ч|:00)', lower)
    if time_match:
        time = int(time_match.group(1)) % 24
    elif 'ноч' in lower:
        time = 2
    elif 'утр' in lower:
        time = 8
    elif 'вечер' in lower:
        time = 20

    return time, amount

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    user_time, user_amount = parse_transaction(text)

    # Передаем данные в модель RandomForest
    new_transaction = pd.DataFrame([[user_time, user_amount]], columns=['Time', 'Amount'])
    probabilities = model.predict_proba(new_transaction)[0]
    fraud_prob = float(probabilities[1] * 100)

    POROG = 15.0
    is_danger = fraud_prob >= POROG

    return jsonify({
        'fraudProbability': round(fraud_prob, 1),
        'verdict': 'danger' if is_danger else 'safe',
        'time': user_time,
        'amount': user_amount
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
