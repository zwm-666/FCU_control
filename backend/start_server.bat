@echo off
echo Starting H2 FCU Backend Server...
echo.
call conda activate base
python -u server.py
