import os
import sys

print("=" * 60)
print("ПРОВЕРКА ПРОЕКТА")
print("=" * 60)

print("\n1. ПРОВЕРКА ПАПОК:")
print(f"Текущая папка: {os.getcwd()}")
print(f"Содержимое: {os.listdir('.')}")

print("\n2. ПРОВЕРКА ДАННЫХ:")
data_path = "data"
if os.path.exists(data_path):
    print(f"Папка 'data' существует")
    files = os.listdir(data_path)
    print(f"Файлы в data/: {files}")
    
    for file in ['train.csv', 'test.csv']:
        if file in files:
            filepath = os.path.join(data_path, file)
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            print(f"  ✅ {file}: {size:.2f} MB")
        else:
            print(f"  ❌ {file}: не найден")
else:
    print("❌ Папка 'data' не существует!")

print("\n3. ПРОВЕРКА БИБЛИОТЕК:")
try:
    import pandas as pd
    print(f"✅ pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import sklearn
    print(f"✅ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn: {e}")

try:
    import catboost
    print(f"✅ catboost: {catboost.__version__}")
except ImportError as e:
    print(f"❌ catboost: {e}")

print("\n4. ПРОВЕРКА PYTHON:")
print(f"Версия Python: {sys.version}")
print(f"Путь к Python: {sys.executable}")

print("\n" + "=" * 60)
print("ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 60)

if os.path.exists("data/train.csv") and os.path.exists("data/test.csv"):
    print("\n🎉 ВСЕ ГОТОВО К ЗАПУСКУ!")
    print("Запустите: python main.py")
else:
    print("\n⚠ ПРОВЕРЬТЕ ДАННЫЕ!")
    print("Убедитесь, что train.csv и test.csv в папке data/")
