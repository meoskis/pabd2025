import os
import joblib
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ===== Настройка логирования в файл рядом со скриптом =====
base_dir = os.path.abspath(os.path.dirname(__file__))  # Папка, где находится скрипт
log_dir = os.path.join(base_dir, 'logs')               # Путь к директории logs
os.makedirs(log_dir, exist_ok=True)                    # Создаём logs/ если её нет

log_path = os.path.join(log_dir, 'app.log')            # Путь к лог-файлу

file_handler = RotatingFileHandler(log_path, maxBytes=10240, backupCount=5)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
file_handler.setFormatter(formatter)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
# ==========================================================

model = joblib.load('./models/linear_regression_model.pkl')
app.logger.info("Загружена ML-модель")

# Маршрут для отображения формы
@app.route('/')
def index():
    app.logger.info("Открыта главная страница")
    return render_template('index.html')

def make_price_prediction(inputs):
    prediction = model.predict(inputs)
    
    return prediction

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    
    try:
        data = request.get_json()
        app.logger.info(f"Получен JSON: {data}")
        required_fields = ['number_of_rooms', 'area', 'flat_floor', 'total_floors']

        # Проверка наличия всех полей
        for field in required_fields:
            if field not in data:
                msg = f"Отсутствует поле: {field}"
                app.logger.warning(msg)
                return jsonify({'status': 'error', 'message': msg})

        # Преобразование и валидация значений
        numbers = []
        for field in required_fields:
            try:
                num = float(data[field])
                if num < 0:
                    msg = f'{field} должно быть положительным'
                    app.logger.warning(msg)
                    return jsonify({'status': 'error', 'message': msg})
                
                if field == 'number_of_rooms' and num > 4:
                    msg = 'Количество комнат не может быть больше четырёх'
                    app.logger.warning(msg)
                    return jsonify({'status': 'error', 'message': msg})
                
                if field == 'flat_floor' and num > float(data['total_floors']):
                    msg = 'Этаж квартиры не может быть больше общего количества этажей'
                    app.logger.warning(msg)
                    return jsonify({'status': 'error', 'message': msg})
                
                if field == 'area' and num > 500:
                    msg = 'Площадь квартиры не может превышать 500 м²'
                    app.logger.warning(msg)
                    return jsonify({'status': 'error', 'message': msg})
                
                numbers.append(num)
            except ValueError:
                msg = f"{field} должно быть числом"
                app.logger.warning(msg)
                return jsonify({'status': 'error', 'message': msg})

        app.logger.info(f"Данные успешно проверены: {numbers}")
        
        area = float(data.get('area'))
        app.logger.info(f"Запуск предсказания для площади: {area} м²")

        prediction = make_price_prediction([[area]])
        app.logger.info(f"Предсказание сделано: {prediction}")
        
        return jsonify({'status': 'success', 'prediction': float(prediction[0])})

    except Exception as e:
        app.logger.exception("Ошибка сервера при обработке запроса")
        return jsonify({'status': 'error', 'message': f'Ошибка сервера: {str(e)}'})

if __name__ == '__main__':
    app.logger.info("Запуск сервера на http://127.0.0.1:5000")
    app.run(debug=True)