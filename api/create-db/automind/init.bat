set Server=localhost
set DBName=AutoML
set Directory=C:\DB\MSSQL
set UID=automl
set Password=.automl.

echo "Initial AutoMind Database Tuple, Function, View, SP"

sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\00_init_system.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\01_1_init_member.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\01_2_init_ml_engine.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\02_1_fn_get_member_cid.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\03_view.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\04_2_sp_data_source.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\04_3_sp_project.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\04_4_sp_model.sql
sqlcmd -S %Server% -E -v DBName = "%DBName%" DirPath = "%Directory%" -i sql\04_5_sp_app.sql

pause
