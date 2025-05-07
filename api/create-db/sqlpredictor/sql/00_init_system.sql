-------------------------------------------------------------------------------
-- Requiremenmt																												    	  |
-------------------------------------------------------------------------------
-- xp_insertClass
USE [AutoML];


GO
-------------------------------------------------------------------------------
-- Initial Entity Type																									    	|
-------------------------------------------------------------------------------
SET
IDENTITY_INSERT [dbo].[Entity] ON;


MERGE INTO
	[dbo].[Entity] AS Target USING (
		VALUES
			(101, '系統目錄', 'system class', 1),
			(102, '會員目錄', 'member class', 1),
			(103, '資料目錄', 'data source class', 1),
			(104, '機器學習引擎目錄', 'ml engine class', 1),
			(105, '模型目錄', 'model class', 1),
			(106, '專案目錄', 'project class', 1),
			(107, '應用目錄', 'app class', 1),
			(108, '儀錶板目錄', 'dashboard', 1),
			(109, '資料:檔案', 'data:file', 1),
			(110, '資料:mssql', 'data:mssql', 1),
			(111, '系統機器學習引擎', 'engine:system', 1),
			(112, '模型', 'model', 1),
			(113, '應用:預測', 'app:prediction', 1),
			(114, '應用:排程', 'app:schedule', 1),
			(115, '機器學習引擎', 'engine:member', 1)
	) AS Source (EID, CName, EName, bORel) ON Target.EName = Source.EName
WHEN MATCHED THEN
UPDATE SET
	CName = Source.CName
WHEN NOT MATCHED BY TARGET THEN
INSERT
	(EID, CName, EName, bORel)
VALUES
	(EID, CName, EName, bORel);


SET
IDENTITY_INSERT [dbo].[Entity] OFF;


GO
-------------------------------------------------------------------------------
-- Initial System Classes																								    	|
-------------------------------------------------------------------------------
SET
IDENTITY_INSERT [dbo].[class] ON;


DECLARE @adminMID int;


SELECT
	@adminMID = (
		SELECT
			MID
		FROM
			Member
		WHERE
			Account = 'admin'
	);


MERGE INTO
	[dbo].[Class] AS Target using (
		VALUES
			(1, 101, 'root', 1, 'root', 0, @adminMID)
	) AS Source (
		CID,
		TYPE,
		CName,
		IDPath,
		NamePath,
		nLevel,
		OwnerMID
	) ON Target.CID = Source.CID
WHEN MATCHED THEN
UPDATE SET
	CName = Source.CName,
	NamePath = Source.NamePath,
	nLevel = Source.nLevel,
	OwnerMID = Source.OwnerMID
WHEN NOT MATCHED THEN
INSERT
	(
		CID,
		TYPE,
		CName,
		IDPath,
		NamePath,
		nLevel,
		OwnerMID
	)
VALUES
	(
		CID,
		TYPE,
		CName,
		IDPath,
		NamePath,
		nLevel,
		OwnerMID
	);


SET
IDENTITY_INSERT [dbo].[class] OFF;


DECLARE @NewCID int;


EXEC xp_insertClass 1,
102,
'member',
'collect member classes',
1,
@NewCID OUTPUT;


EXEC xp_insertClass 1,
104,
'ml_engine',
'collect system''s ml engines',
1,
@NewCID OUTPUT;


SET
NOEXEC OFF;


GO
