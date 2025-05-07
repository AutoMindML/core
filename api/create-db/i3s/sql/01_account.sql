-------------------------------------------------------------------------------
-- Create account																												    	|
-------------------------------------------------------------------------------
USE [$(DBName)];


GO
-- Notice:
-- Add login don't need to select DB, because it's global,
-- but add user need to select DB, it's for alter user of DB
DECLARE @uid varchar(20),
@pwd varchar(20),
@db varchar(20)
SELECT
	@uid = '$(UID)',
	@pwd = '$(Password)',
	@db = '$(DBName)';


-- Add default login
IF NOT EXISTS (
	SELECT
		*
	FROM
		master.dbo.syslogins
	WHERE
		name = @uid
) BEGIN EXEC sp_addlogin @uid,
@pwd,
@db END ELSE PRINT 'the user was already existed' EXEC sp_adduser @loginame = @uid,
@name_in_db = @uid,
@grpname = 'db_owner';


GO
-------------------------------------------------------------------------------
-- Alter rule											      																    	|
-------------------------------------------------------------------------------
ALTER
ROLE db_owner
ADD member [$(UID)];


GO
-------------------------------------------------------------------------------
-- Connect user to login													                    				|
-------------------------------------------------------------------------------
ALTER
USER [$(UID)]
WITH
LOGIN = [$(UID)];
