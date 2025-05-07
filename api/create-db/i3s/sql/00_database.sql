CREATE
DATABASE [$(DBName)] ON (
	NAME = '$(DBName)_data',
	FILENAME = '$(DirPath)\$(DBName)_data.mdf',
	SIZE = 10,
	FILEGROWTH = 5
) LOG ON (
	NAME = '$(DBName)_log',
	FILENAME = '$(DirPath)\$(DBName)_log.ldf',
	SIZE = 10,
	FILEGROWTH = 5
);
