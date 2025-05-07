USE [AutoML];


GO
-------------------------------------------------------------------------------
-- SP: create member class and views																	    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	xp_createMemberClass @mid int AS BEGIN
	--
	DECLARE @memberClass int,
	@account varchar(100),
	@memberNewCID int,
	@newCID int;


SELECT
	@memberClass = (
		SELECT
			CID
		FROM
			Class
		WHERE
			NamePath = 'member'
			AND
		TYPE = 102
	);


SELECT
	@account = (
		SELECT
			Account
		FROM
			Member
		WHERE
			MID = @mid
	);


EXEC xp_insertClass @memberClass,
102,
@account,
NULL,
@mid,
@memberNewCID OUTPUT;


EXEC xp_insertClass @memberNewCID,
103,
'data_source',
NULL,
@mid,
@newCID OUTPUT;


EXEC xp_insertClass @memberNewCID,
104,
'ml_engine',
NULL,
@mid,
@newCID OUTPUT;


--EXEC xp_insertClass @memberNewCID,
--105,
--'model',
--NULL,
--@mid,
--@newCID OUTPUT;
EXEC xp_insertClass @memberNewCID,
106,
'project',
NULL,
@mid,
@newCID OUTPUT;


--EXEC xp_insertClass @memberNewCID,
--107,
--'app',
--NULL,
--@mid,
--@newCID OUTPUT;
EXEC xp_insertClass @memberNewCID,
108,
'dashboard',
NULL,
@mid,
@newCID OUTPUT;


END;


GO
-------------------------------------------------------------------------------
-- SP: delete member class																						    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	xp_deleteMemberClass @mid int AS BEGIN
	--
	DECLARE @account varchar(100);


SELECT
	@account = (
		SELECT
			Account
		FROM
			Member
		WHERE
			MID = @mid
	);


DECLARE classCursor CURSOR FOR (
	SELECT
		CID
	FROM
		[dbo].[Class]
	WHERE
		NamePath LIKE 'member/' + @account
		OR NamePath LIKE 'member/' + @account + '/%'
);


OPEN classCursor;


DECLARE @cid int;


FETCH NEXT
FROM
	classCursor
INTO
	@cid WHILE (@@FETCH_STATUS <> -1) BEGIN
	--
	EXEC xp_deleteClass @cid;


FETCH NEXT
FROM
	classCursor
INTO
	@cid;


END;


CLOSE classCursor;


DEALLOCATE classCursor;


END;


GO
-------------------------------------------------------------------------------
-- Check and fix all member class																			    	  |
-------------------------------------------------------------------------------
-- Find member that doesn't has own class
DECLARE memberCursor CURSOR FOR (
	SELECT
		M.MID
	FROM
		[dbo].[Member] M
		LEFT JOIN [dbo].[Class] C ON C.NamePath = 'member/' + M.Account
	WHERE
		CID IS NULL
		AND M.Account != 'admin'
);


OPEN memberCursor;


DECLARE @mid int;


FETCH NEXT
FROM
	MemberCursor
INTO
	@mid WHILE (@@FETCH_STATUS <> -1) BEGIN
	--
	EXEC xp_createMemberClass @mid;


FETCH NEXT
FROM
	MemberCursor
INTO
	@mid;


END;


CLOSE memberCursor;


DEALLOCATE memberCursor;


GO
