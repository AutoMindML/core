CREATE
FUNCTION checkpermission_subclass (@CID int, @MID int) returns @CT
TABLE (CID int) AS BEGIN DECLARE @ID int IF NOT EXISTS (
	SELECT
		permissionbits
	FROM
		Permission
	WHERE
		cid = @CID
		AND roletype = CAST(1 AS bit)
		AND roleid = @MID
		AND PermissionBits & 7 = 7
) BEGIN IF EXISTS (
	SELECT
		cid
	FROM
		permission
	WHERE
		cid = @CID
		AND roletype = CAST(0 AS bit)
		AND roleid IN (
			SELECT
				gid
			FROM
				gm
			WHERE
				mid = @MID
		)
		AND permissionbits & 7 = 7
) BEGIN DECLARE cidcursor CURSOR FOR
SELECT DISTINCT
	cid
FROM
	permission
WHERE
	cid IN (
		SELECT
			ccid
		FROM
			inheritance
		WHERE
			pcid = @CID
	)
	AND roletype = CAST(0 AS bit)
	AND roleid IN (
		SELECT
			gid
		FROM
			gm
		WHERE
			mid = @MID
	)
	AND permissionbits & 1 = 1 OPEN cidcursor
FETCH NEXT
FROM
	cidcursor
INTO
	@ID WHILE (@@FETCH_STATUS <> -1) BEGIN
INSERT INTO
	@CT
VALUES
	(@ID)
FETCH NEXT
FROM
	cidcursor
INTO
	@ID END;


CLOSE cidcursor DEALLOCATE cidcursor END;


END ELSE BEGIN DECLARE cidcursor CURSOR FOR
SELECT
	cid
FROM
	permission
WHERE
	cid IN (
		SELECT
			ccid
		FROM
			inheritance
		WHERE
			pcid = @CID
	)
	AND roletype = CAST(1 AS bit)
	AND roleid = @MID
	AND permissionbits & 1 = 1 OPEN cidcursor
FETCH NEXT
FROM
	cidcursor
INTO
	@ID WHILE (@@FETCH_STATUS <> -1) BEGIN
INSERT INTO
	@CT
VALUES
	(@ID)
FETCH NEXT
FROM
	cidcursor
INTO
	@ID END;


CLOSE cidcursor DEALLOCATE cidcursor END;


RETURN END;


GO
--------------------------------------------------------------------------------------------------------
-- select * from class where cid in(select * from checkpermission_subclass(0,1))-------------------------
--------------------------------------------------------------------------------------------------------
CREATE
FUNCTION checkpermission_objlist (@CID int, @MID int) returns @CT
TABLE (CID int) AS BEGIN IF EXISTS (
	SELECT
		permissionbits
	FROM
		permission
	WHERE
		cid = @CID
		AND roletype = CAST(1 AS bit)
		AND roleid = @MID
		AND permissionbits & 7 = 7
) BEGIN
INSERT INTO
	@CT
VALUES
	(@CID) END;


ELSE IF EXISTS (
	SELECT
		permissionbits
	FROM
		permission
	WHERE
		cid = @CID
		AND roletype = CAST(0 AS bit)
		AND roleid IN (
			SELECT
				gid
			FROM
				gm
			WHERE
				mid = @MID
		)
		AND permissionbits & 7 = 7
) BEGIN
INSERT INTO
	@CT
VALUES
	(@CID) END;


RETURN END;


GO
--------------------------------------------------------------------------------------------------------
--select * from class where cid in(select * from checkpermission_objlist(0,1));-------------------------
--------------------------------------------------------------------------------------------------------
/*----------------------------------------------------------------------------*/
CREATE
FUNCTION getPassportcodeMember (@passportcode nvarchar(32)) returns @MT
TABLE (MID int) AS BEGIN DECLARE @ID int IF @passportcode IS NULL BEGIN DECLARE midcursor CURSOR FOR
SELECT
	mid
FROM
	member
WHERE
	Account = 'Guest' OPEN midcursor
FETCH NEXT
FROM
	midcursor
INTO
	@ID IF (@@FETCH_STATUS <> -1) BEGIN
INSERT INTO
	@MT
VALUES
	(@ID) END;


CLOSE midcursor DEALLOCATE midcursor END;


ELSE BEGIN DECLARE midcursor2 CURSOR FOR
SELECT
	mid
FROM
	msession
WHERE
	passportcode LIKE @passportcode OPEN midcursor2
FETCH NEXT
FROM
	midcursor2
INTO
	@ID IF (@@FETCH_STATUS <> -1) BEGIN
INSERT INTO
	@MT
VALUES
	(@ID) END;


ELSE BEGIN DECLARE midcursor3 CURSOR FOR
SELECT
	mid
FROM
	member
WHERE
	Account = 'Guest' OPEN midcursor3
FETCH NEXT
FROM
	midcursor3
INTO
	@ID IF (@@FETCH_STATUS <> -1) BEGIN
INSERT INTO
	@MT
VALUES
	(@@FETCH_STATUS)
INSERT INTO
	@MT
VALUES
	(@@FETCH_STATUS) END;


CLOSE midcursor3 DEALLOCATE midcursor3 END;


CLOSE midcursor2 DEALLOCATE midcursor2 END;


RETURN END;


GO
/*----------------------------------------------------------------------------*/
/* CData Table*/
CREATE TABLE CData (
	DID int identity(1, 1) NOT NULL,
	DName nvarchar(255) NULL,
	NSpace nvarchar(20) NULL,
	Condition nvarchar(800) NULL,
	LimitNum smallint NULL,
	Sort nvarchar NULL,
	SortDesc bit NULL DEFAULT (1),
	PermissionLevel smallint NOT NULL DEFAULT (0),
	Name nvarchar(255) NULL,
	Des nvarchar(800) NULL,
	CONSTRAINT PK_CData PRIMARY KEY CLUSTERED (DID)
);


GO
/*----------------------------------------------------------------------------*/
INSERT INTO
	CData (DName, NSpace, PermissionLevel)
VALUES
	('vd_subclass', 'Dublin Core', 2);


/*----------------------------------------------------------------------------*/
INSERT INTO
	CLayout (LName, LDes)
VALUES
	('i3s-awesome', 'i3s awesome index page');


/*----------------------------------------------------------------------------*/
UPDATE class
SET
	Layout = 1
WHERE
	cid IN (1);


;


GO
/*----------------------------------------------------------------------------*/
CREATE VIEW vs_ClassLayout AS
SELECT
	c.CID AS CID,
	l.LID AS LID,
	l.LName AS Layout
FROM
	Class c,
	CLayout l
WHERE
	c.Layout = l.LID;


;


GO
/*----------------------------------------------------------------------------*/
INSERT INTO
	gm
VALUES
	(1, 0, 1, 1, 1);


GO
/*----------------------------------------------------------------------------*/
INSERT INTO
	cdata (dname, NSpace, PermissionLevel)
VALUES
	('vd_ObjectList', 'Dublin Core', 1),
	('vd_ShowObject', 'Dublin Core', 0);
