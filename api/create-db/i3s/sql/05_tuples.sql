USE [$(DBName)];


GO
/*
部署後指令碼樣板							
--------------------------------------------------------------------------------------
此檔案包含要附加到組建指令碼的 SQL 陳述式		
使用 SQLCMD 語法可將檔案包含在部署後指令碼中			
範例:      :r .\myfile.sql								
使用 SQLCMD 語法可參考部署後指令碼中的變數		
範例:      :setvar TableName MyTable							
SELECT * FROM [$(TableName)]					
--------------------------------------------------------------------------------------
*/
MERGE INTO
	Entity AS Target USING (
		VALUES
			('WEB資源', 'URL', 1),
			('會員', 'Member', 1),
			('檔案', 'Archive', 1),
			('公告', 'Announce', 1)
	) AS Source (CName, EName, bORel) ON Target.EName = Source.EName
WHEN MATCHED THEN
UPDATE SET
	CName = Source.CName
WHEN NOT MATCHED BY TARGET THEN
INSERT
	(CName, EName, bORel)
VALUES
	(CName, EName, bORel);


MERGE INTO
	SystemConfig AS Target USING (
		VALUES
			('ExpiredDT', 60),
			('SingleSignOn', 0)
	) AS Source (Name, Des) ON Target.Name = Source.Name
WHEN MATCHED THEN
UPDATE SET
	Des = Source.Des
WHEN NOT MATCHED BY TARGET THEN
INSERT
	(Name, Des)
VALUES
	(Name, Des);


MERGE INTO
	Groups AS Target USING (
		VALUES
			('Administrators', 'System Administrator Group'),
			('Users', 'System User Group')
	) AS Source (GName, GDes) ON Target.GName = Source.GName
WHEN MATCHED THEN
UPDATE SET
	GName = Source.GName
WHEN NOT MATCHED BY TARGET THEN
INSERT
	(GName, GDes)
VALUES
	(GName, GDes);


IF NOT EXISTS (
	SELECT
		*
	FROM
		SystemConfig
	WHERE
		Name = 'License'
) BEGIN
INSERT INTO
	SystemConfig (Name, Des)
VALUES
	('License', NEWID()) END IF NOT EXISTS (
		SELECT
			*
		FROM
			Member
		WHERE
			Account = 'admin'
	) BEGIN DECLARE @mid int EXEC xps_addNewMember 'admin',
	'wke123456',
	'管理員',
	'admin@wke',
	1,
	NULL,
	NULL,
	NULL,
	@mid OUTPUT END;


GO
