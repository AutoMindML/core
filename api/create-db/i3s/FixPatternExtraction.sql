-- setting path of target assembly file
DECLARE @dll_path nvarchar(500) = '$(Directory)\PatternExtraction.dll';


-- get name of current database
DECLARE @DBName nvarchar(50) = DB_NAME()
-- setting TRUSTWORTHY, AUTHORIZATION
EXEC (
	'ALTER DATABASE ' + @DBName + ' SET TRUSTWORTHY ON'
) EXEC (
	'ALTER AUTHORIZATION ON database::' + @DBName + ' TO sa;'
);


-- drop function if already exists
DROP
FUNCTION IF EXISTS dbo.fn_RegexMatch;


DROP
ASSEMBLY IF EXISTS RegExp
-- enable CLR
EXEC sp_configure 'clr enabled',
'1';


RECONFIGURE;


-- create assembly
CREATE
ASSEMBLY RegExp
FROM
	@dll_path
WITH
	permission_set = external_access;


-- create function for runing target assembly
CREATE
OR ALTER
FUNCTION fn_RegexMatch (@Pattern nvarchar(MAX), @Text nvarchar(MAX)) returns
TABLE (
	SNO int,
	POSITION int,
	Words nvarchar(MAX),
	LEN int,
	nGroup int
)
WITH
EXECUTE AS caller AS EXTERNAL name RegExp.UDF_RegExp.RegExp;


-- testing query -- find all number text
--select * from fn_RegexMatch('\d*', 'ABC123DEF456GHI78JKL9')
