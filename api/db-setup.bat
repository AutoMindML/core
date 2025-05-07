cd %~dp0\create-db\i3s
call create-sqlpredictor.bat

cd %~dp0\create-db\sqlpredictor
call init-sqlpredictor.bat

echo "Finished!!!"
pause
