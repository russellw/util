if not exist C:\Users\User\Documents\lumagraph.db exit /b
dir C:\Users\User\Documents\lumagraph.db 
sqlite3 C:\Users\User\Documents\lumagraph.db "SELECT * FROM folders;"
sqlite3 C:\Users\User\Documents\lumagraph.db "SELECT ID, file, name, folder, added FROM documents;"
