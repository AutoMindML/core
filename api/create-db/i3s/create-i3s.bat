set Server=localhost
set DBName=I3S
set Directory=C:\DB\MSSQL
set UID=wke
set Password=.wke.

echo "copy dll files"
copy *.dll %Directory%
pause

echo "create DB"
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\00_database.sql
pause

echo "create account"
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\01_account.sql
pause

echo "create tables"
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\02_1_i3s_table.sql
pause

echo "create views"
sqlcmd -S %Server% -E -v DBName = "%DBName%" -i sql\03_1_i3s_view.sql
pause

echo "create sp and function"
sqlcmd -S %Server% -E -v DBName = "%DBName%" DLLDir = "%Directory%" -i sql\04_sp_func.sql
pause

echo "insert tuples"
sqlcmd -S %Server% -E -v DBName = "%DBName%" DLLDir = "%Directory%" -i sql\05_tuples.sql
pause
