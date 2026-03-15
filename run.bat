@echo off
title Trading Terminal
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo  Trading Terminal starting...
echo  Dashboard: http://localhost:8501
echo.
streamlit run dashboard/app.py
pause
