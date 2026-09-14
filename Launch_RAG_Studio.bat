@echo off
title PyTorch Transformer RAG Studio (Port 4881)
cd /d "%~dp0"
echo ========================================================
echo Starting PyTorch Local Transformer RAG Studio on Port 4881...
echo ========================================================
start "" "http://localhost:4881/"
python app.py
pause
