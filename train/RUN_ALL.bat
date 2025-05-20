@echo off

echo Запуск train_mt5.py
python train_mt5.py || exit /b

echo Запуск train_deberta.py
python train_deberta.py || exit /b

echo Запуск train_xlmr.py
python train_xlmr.py || exit /b

echo Готово!
pause
