@echo off
cd /d "c:\Users\cakyildirim\PycharmProjects\RL-Trading-Project"
venv\Scripts\python.exe tests\smoke_test_source.py > tests\smoke_out.txt 2> tests\smoke_err.txt
echo Exit code: %ERRORLEVEL%
