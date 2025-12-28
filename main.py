"""
Основной файл с решением соревнования
Здесь весь код для создания предсказаний и формирования submission.csv
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


def create_submission(predictions: pd.DataFrame) -> str:
    """
    Сохраняет предсказания в файл results/submission.csv
    ВНИМАНИЕ: файл должен называться именно submission.csv и лежать в папке results
    """
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    print(f"Submission файл сохранён: {submission_path}")
    return submission_path


def main():
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    # Фиксируем seed для воспроизводимости
    np.random.seed(322)

    # Путь к данным (в Docker-контейнере данные будут в текущей директории)
    # Важно: в Docker файлы train.csv, test.csv и sample_submission.csv лежат в корне проекта
    DATA_PATH = './'  # в контейнере — текущая папка

    # Загружаем данные
    train = pd.read_csv(DATA_PATH + 'train.csv', parse_dates=['dt'])
    test = pd.read_csv(DATA_PATH + 'test.csv', parse_dates=['dt'])
    sample_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    # Сортируем train по дате
    train = train.sort_values('dt').reset_index(drop=True)

    # Категориальные и числовые признаки
    cat_cols = [
        'product_id', 'management_group_id', 'first_category_id',
        'second_category_id', 'third_category_id',
        'dow', 'month', 'holiday_flag', 'activity_flag'
    ]
    num_cols = [
        'day_of_month', 'week_of_year', 'n_stores',
        'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level'
    ]
    features = cat_cols + num_cols

    # Логарифмируем целевые переменные
    train['price_p05_log'] = np.log1p(train['price_p05'])
    train['price_p95_log'] = np.log1p(train['price_p95'])

    # Label Encoding категориальных признаков
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        # Для теста: если значение не встречалось — ставим -1
        test[col] = test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    # Подготовка данных для обучения
    X = train[features]
    y_p05 = train['price_p05_log']
    y_p95 = train['price_p95_log']
    X_test = test[features]

    # Параметры LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 322,
        'n_jobs': -1
    }

    # Обучаем модель для price_p05
    print("Обучаем модель для price_p05...")
    model_p05 = lgb.train(
        params,
        lgb.Dataset(X, label=y_p05),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X, label=y_p05)],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
    )

    # Обучаем модель для price_p95
    print("Обучаем модель для price_p95...")
    model_p95 = lgb.train(
        params,
        lgb.Dataset(X, label=y_p95),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X, label=y_p95)],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
    )

    # Предсказания
    pred_p05_log = model_p05.predict(X_test)
    pred_p95_log = model_p95.predict(X_test)

    pred_p05 = np.expm1(pred_p05_log)
    pred_p95 = np.expm1(pred_p95_log)

    # Гарантируем p05 ≤ p95
    mask = pred_p05 > pred_p95
    pred_p05[mask], pred_p95[mask] = pred_p95[mask].copy(), pred_p05[mask].copy()

    # Округляем до 2 знаков
    pred_p05 = np.round(pred_p05, 2)
    pred_p95 = np.round(pred_p95, 2)

    # Формируем submission
    submission = sample_sub.copy()
    submission['price_p05'] = pred_p05
    submission['price_p95'] = pred_p95

    # Сохраняем через обязательную функцию
    create_submission(submission)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()