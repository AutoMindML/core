USE [AutoML];


GO
-------------------------------------------------------------------------------
-- SP: Add ML Engine																										    	|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_add_ml_engine] @mid int,
	@name nvarchar(512),
	@des nvarchar(4000),
	@handler nvarchar(50),
	@params nvarchar(MAX),
	@newOID int OUTPUT AS BEGIN
INSERT INTO
	[dbo].[Object] (
		CName,
		CDes,
		TYPE,
		DataByte,
		OwnerMID
	)
VALUES
	(@name, @des, 111, 0, @mid);


SELECT
	@newOID = SCOPE_IDENTITY();


DECLARE @md5 binary(16) = HASHBYTES('MD5', CONCAT(@handler, @params));


IF NOT EXISTS (
	SELECT
		MD5
	FROM
		ML_Engine
	WHERE
		MD5 = @md5
)
INSERT INTO
	ML_Engine (MLEID, MD5, Handler, ConnectionData)
VALUES
	(@newOID, @md5, @handler, @params);


UPDATE Object
SET
	EName = convert(varchar(128), @md5, 2)
WHERE
	OID = @newOID;


DECLARE @system_ml_engine_cid int = (
	SELECT
		CID
	FROM
		Class
	WHERE
		NamePath = 'ml_engine'
);


INSERT INTO
	CO (CID, OID)
VALUES
	(@system_ml_engine_cid, @newOID);


END;


GO
-------------------------------------------------------------------------------
-- Initial ML Engine																											  	|
-------------------------------------------------------------------------------
DECLARE @newOID int;


IF NOT EXISTS (
	SELECT
		*
	FROM
		Object O
		INNER JOIN ML_Engine E ON O.OID = E.MLEID
	WHERE
		O.Type = 111
		AND E.Handler = 'lightwood'
) BEGIN EXEC [dbo].[xp_add_ml_engine] 1,
'lightwood',
'Lightwood is the default AI engine used in MindsDB. It deals mainly with classification, regression, and time-series problems in machine learning.',
'lightwood',
'{}',
@newOID OUTPUT;


END;

---

IF NOT EXISTS (
	SELECT
		*
	FROM
		Object O
		INNER JOIN ML_Engine E ON O.OID = E.MLEID
	WHERE
		O.Type = 111
		AND E.Handler = 'tpot'
) BEGIN EXEC [dbo].[xp_add_ml_engine] 1,
'tpot',
'A Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.',
'tpot',
'{}',
@newOID OUTPUT;


END;

GO
