if not exist C:\Users\User\Documents\lumagraph.db exit /b
sqlite3 C:\Users\User\Documents\lumagraph.db .dump|head -n 30
