import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Настройка внешнего вида сайта (заголовок вкладки и иконка)
st.set_page_config(page_title="Антифрод ИИ", page_icon="🛡️", layout="centered")

# Функция для обучения ИИ (кэшируем её, чтобы сайт не перезапускал обучение при каждом клике)
@st.cache_resource
def load_and_train_ai():
    df = pd.read_csv('creditcard.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Загружаем обученную модель
model = load_and_train_ai()

# 2. Оформление веб-страницы
st.title("🛡️ Интеллектуальная система Антифрод")
st.write("Проект: Искусственный интеллект в борьбе с финансовым мошенничеством")
st.markdown("---")

st.subheader("📊 Проверка новой транзакции")

# Создаем красивую форму ввода прямо на сайте
with st.form("fraud_form"):
    # Ползунок для выбора времени от 0 до 23
    user_time = st.slider("Время совершения операции (час)", min_value=0, max_value=23, value=12)
    
    # Поле для ввода суммы числом
    user_amount = st.number_input("Сумма транзакции", min_value=0.0, max_value=100000.0, value=150.0, step=10.0)
    
    # Кнопка отправки формы на проверку
    submit_button = st.form_submit_button(label="Проверить транзакцию")

# 3. Логика работы ИИ при нажатии на кнопку
if submit_button:
    # Оформляем данные пользователя в таблицу для ИИ
    new_transaction = pd.DataFrame([[user_time, user_amount]], columns=['Time', 'Amount'])
    
    # Получаем вероятность мошенничества
    probabilities = model.predict_proba(new_transaction)
    fraud_probability = probabilities[0][1] * 100 # Берем вероятность класса 1 (фрод)
    
    POROG = 15.0 # Наш проверенный порог безопасности
    
    st.markdown("### 📈 Вердикт антифрод-системы:")
    st.metric(label="Вероятность мошенничества", value=f"{fraud_probability:.1f}%")
    
    if fraud_probability >= POROG:
        st.error("🚨 ВНИМАНИЕ: ОБНАРУЖЕНО МОШЕННИЧЕСТВО!")
        st.warning("🛑 Статус: Транзакция ЗАБЛОКИРОВАНА в целях безопасности.")
        st.info(f"📋 Причина: Высокий уровень риска (выше критического порога {POROG}%)")
    else:
        st.success("✅ ТРАНЗАКЦИЯ ОДОБРЕНА")
        st.info("💰 Статус: Платеж успешно проведен. Риски не обнаружены.")
