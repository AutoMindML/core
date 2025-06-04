set Server=localhost
set DBName=AutoML
set Directory=C:\DB\MSSQL
set UID=automl
set Password=.automl.

copy *.dll %Directory%

echo "Create AutoMind Database with I3S Schema"

sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\00_database.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\01_account.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\02_2_automl_table.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" -i sql\03_1_i3s_view.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DLLDir = "%Directory%" -i sql\04_sp_func.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DLLDir = "%Directory%" -i sql\05_tuples.sql

pause
