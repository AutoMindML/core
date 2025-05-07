USE [$(DBName)];


GO
--啟用CLR功能
ALTER
DATABASE [$(DBName)]
SET
	TRUSTWORTHY ON;


-- required in MSSQL 2017
EXEC sp_configure 'clr enable',
'1';


RECONFIGURE
WITH
	OVERRIDE;


GO
--Create Assembly
IF EXISTS (
	SELECT
		*
	FROM
		sys.assemblies asms
	WHERE
		asms.name = N'BitwiseOperatorsCLRFunc'
) DROP
ASSEMBLY BitwiseOperatorsCLRFunc;


CREATE
ASSEMBLY BitwiseOperatorsCLRFunc
FROM
	'$(DLLDir)\BitwiseOperators.dll';


;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_checkBitwise]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_checkBitwise] (@BitValue bigint, @BitPosition tinyint) returns bit AS BEGIN DECLARE @Compare bigint = (power(convert(bigint, 2), @BitPosition)) DECLARE @Output bit = 0 IF (
	(
		SELECT
			@BitValue & @Compare
	) = @Compare
)
SET
	@Output = 1 ELSE
SET
	@Output = 0 RETURN @Output END;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_checkUserPermission]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_checkUserPermission] (@CID int, @MID int, @BitPosition tinyint) returns bit AS BEGIN DECLARE @Result bit = 0 DECLARE @Count int = (
	SELECT
		count(*)
	FROM
		Permission
	WHERE
		CID = @CID
		AND RoleType = 1
		AND RoleID = @MID
) IF (@Count = 1) BEGIN
SET
	@Result = (
		SELECT
			dbo.fs_checkBitwise (PermissionBits, @BitPosition)
		FROM
			Permission
		WHERE
			CID = @CID
			AND RoleType = 1
			AND RoleID = @MID
	) END ELSE BEGIN DECLARE @Sum int = 0
SET
	@Sum = (
		SELECT
			sum(
				convert(
					int,
					dbo.fs_checkBitwise (p.PermissionBits, @BitPosition)
				)
			)
		FROM
			Permission p,
			GM,
			Groups g
		WHERE
			p.CID = @CID
			AND p.RoleType = 0
			AND p.RoleID = gm.GID
			AND gm.MID = @MID
			AND gm.GID = g.GID
			AND g.bDel = 0
	) IF (@Sum > 0)
SET
	@Result = 1 END RETURN @Result END;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_createPassportCode]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_createPassportCode] (@MID int, @AdditionalString nvarchar(MAX) = '') returns nvarchar(300) AS BEGIN DECLARE @PassportCode nvarchar(300),
@License nvarchar(MAX) = (
	SELECT
		Des
	FROM
		SystemConfig
	WHERE
		Name = 'License'
),
@_MID int,
@_Account int,
@_LoginCount int,
@_Date nvarchar(100) DECLARE @String nvarchar(MAX) = (
	SELECT
		convert(nvarchar(MAX), MID) + Account + convert(nvarchar(MAX), LoginCount) + convert(nvarchar(100), getdate(), 126) + @License + @AdditionalString
	FROM
		Member
	WHERE
		MID = @MID
) DECLARE @i int = 1
SET
	@PassportCode = dbo.fs_getSHA2_512Encode (@String) DECLARE @Count int = (
		SELECT
			count(*)
		FROM
			MSession
		WHERE
			PassportCode = @PassportCode
	) WHILE (@Count = 1) BEGIN
SET
	@String = (
		@PassportCode + convert(nvarchar(100), getdate(), 126) + @i
	)
SET
	@PassportCode = dbo.fs_getSHA2_512Encode (@String)
SET
	@Count = (
		SELECT
			count(*)
		FROM
			MSession
		WHERE
			PassportCode = @PassportCode
	)
SET
	@i = @i + 1 END RETURN @PassportCode END;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_getIDPath]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_getIDPath] (@PCID int, @CID int) returns nvarchar(900) AS BEGIN DECLARE @nLevel int = (
	SELECT
		nLevel
	FROM
		Class
	WHERE
		CID = @PCID
) DECLARE @IDPath nvarchar(900) IF (@nLevel = 0)
SET
	@IDPath = convert(nvarchar(MAX), @CID) ELSE
SET
	@IDPath = (
		SELECT
			IDPath + '/' + convert(nvarchar(MAX), @CID)
		FROM
			Class
		WHERE
			CID = @PCID
	) RETURN @IDPath END;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_getMD5Encode]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_getMD5Encode] (@String nvarchar(MAX)) returns nvarchar(32) AS BEGIN RETURN (
	SELECT
		convert(nvarchar(32), hashbytes('MD5', @String), 2)
) END;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_getNamePath]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_getNamePath] (@PCID int, @CName nvarchar(255)) returns nvarchar(900) AS BEGIN DECLARE @nLevel int = (
	SELECT
		nLevel
	FROM
		Class
	WHERE
		CID = @PCID
) DECLARE @NamePath nvarchar(900) IF (@nLevel = 0)
SET
	@NamePath = @CName ELSE
SET
	@NamePath = (
		SELECT
			NamePath + '/' + @CName
		FROM
			Class
		WHERE
			CID = @PCID
	) RETURN @NamePath END;


GO
/****** Object:  UserDefinedFunction [dbo].[fs_getSHA2_512Encode]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
--
CREATE
OR ALTER
FUNCTION [dbo].[fs_getSHA2_512Encode] (@String nvarchar(MAX)) returns nvarchar(128) AS BEGIN RETURN (
	SELECT
		convert(nvarchar(128), hashbytes('SHA2_512', @String), 2)
) END;


GO
/****** Object:  StoredProcedure [dbo].[xp_insertArchive]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE PROCEDURE
	[dbo].[xp_insertArchive] @FileName nvarchar(255),
	@FileExtension nvarchar(100),
	@ContentLen int,
	@ContentType nvarchar(100),
	@MID int,
	@NewOID int OUTPUT,
	@NewUUID varchar(100) OUTPUT AS BEGIN BEGIN TRY BEGIN TRAN insertArchive DECLARE @EID int = (
		SELECT
			EID
		FROM
			Entity
		WHERE
			EName = 'Archive'
	) IF (@EID IS NULL) RETURN;


DECLARE @CTID int = (
	SELECT
		CTID
	FROM
		ContentType
	WHERE
		Title = @ContentType
) IF (@CTID IS NULL) BEGIN
INSERT INTO
	ContentType (Title)
VALUES
	(@ContentType)
SET
	@CTID = SCOPE_IDENTITY() END
INSERT INTO
	Object (
		TYPE,
		CName,
		OwnerMID
	)
VALUES
	(@EID, @FileName, @MID)
SET
	@NewOID = SCOPE_IDENTITY()
SET
	@NewUUID = NEWID()
INSERT INTO
	Archive (
		AID,
		FileName,
		FileExtension,
		ContentLen,
		ContentType,
		MD5,
		UUID
	)
VALUES
	(
		@NewOID,
		@FileName,
		@FileExtension,
		@ContentLen,
		@CTID,
		hashbytes(
			'MD5',
			CONVERT(varchar(10), @NewOID) + CONVERT(varchar(10), @CTID) + @FileName
		),
		@NewUUID
	) COMMIT TRAN insertArchive END TRY BEGIN CATCH ROLLBACK TRAN insertArchive END CATCH;


END;


GO
/****** Object:  StoredProcedure [dbo].[xp_insertClass]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE PROCEDURE
	[dbo].[xp_insertClass] @PCID int, --c.CID，父節點
	@Type int, --c.Type
	@CName nvarchar(255), --c.Cname
	@CDes nvarchar(4000), --c.CDes
	@OwnerMID int, --m.Member
	@NewCID int OUTPUT --c.CID,新產生的節點，回傳
	AS BEGIN
SET
	@NewCID = NULL DECLARE @_PCID int,
	@_PNamePath nvarchar(900)
SELECT
	@_PCID = CID,
	@_PNamePath = NamePath
FROM
	Class
WHERE
	CID = @PCID BEGIN try IF (@_PCID IS NOT NULL) BEGIN DECLARE @_NewNamePath nvarchar(900) = (
		SELECT
			dbo.fs_getNamePath (@PCID, @CName)
	) IF (
		(
			SELECT
				count(*)
			FROM
				Class
			WHERE
				NamePath = @_NewNamePath
		) = 0
	) BEGIN DECLARE @_NewCID int
INSERT INTO
	Class (CName, [Type], CDes, OwnerMID)
VALUES
	(@CName, @Type, @CDes, @OwnerMID)
SET
	@_NewCID = @@IDENTITY
INSERT INTO
	Inheritance (PCID, CCID)
VALUES
	(@PCID, @_NewCID)
INSERT INTO
	Permission
SELECT
	@_NewCID,
	RoleType,
	RoleID,
	PermissionBits
FROM
	Permission
WHERE
	CID = @PCID DECLARE @_NewLevel int = (
		SELECT
			nLevel + 1
		FROM
			Class
		WHERE
			CID = @PCID
	)
UPDATE Class
SET
	nLevel = @_NewLevel,
	NamePath = @_NewNamePath,
	IDPath = dbo.fs_getIDPath (@PCID, @_NewCID)
WHERE
	CID = @_NewCID
SET
	@NewCID = @_NewCID END ELSE BEGIN RAISERROR ('Error: 節點已存在，不得再新增', 10, 1) RETURN END END ELSE BEGIN RAISERROR ('Error: 沒有PCID', 10, 1) RETURN END END try BEGIN catch
	--系統拋回訊息用
	DECLARE @ErrorMessage AS VARCHAR(1000) = CHAR(10) + '錯誤代碼：' + CAST(ERROR_NUMBER() AS VARCHAR) + CHAR(10) + '錯誤訊息：' + ERROR_MESSAGE() + CHAR(10) + '錯誤行號：' + CAST(ERROR_LINE() AS VARCHAR) + CHAR(10) + '錯誤程序名稱：' + ISNULL(ERROR_PROCEDURE(), '') DECLARE @ErrorSeverity AS Numeric = ERROR_SEVERITY() DECLARE @ErrorState AS Numeric = ERROR_STATE() RAISERROR (@ErrorMessage, @ErrorSeverity, @ErrorState);


--回傳錯誤資訊
RETURN END catch END;


GO
/****** Object:  StoredProcedure [dbo].[xps_addNewMember]    Script Date: 2024/12/3 下午 02:17:43 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
CREATE PROCEDURE
	[dbo].[xps_addNewMember] @Account nvarchar(100),
	@Password nvarchar(256),
	@UserName nvarchar(50),
	@EMail nvarchar(256),
	@Sex bit,
	@Birthday date,
	@Phone nvarchar(100),
	@Address nvarchar(256),
	@NewMID int OUTPUT AS BEGIN BEGIN try IF (
		(
			SELECT
				count(*)
			FROM
				Member
			WHERE
				Account = ltrim(rtrim(@Account))
		) > 0
	) BEGIN RAISERROR ('Error: Account已存在，不可再新增', 10, 1) RETURN END DECLARE @License nvarchar(512) = (
		SELECT
			Des
		FROM
			SystemConfig
		WHERE
			Name = 'License'
	) DECLARE @PWD nvarchar(512) = (
		SELECT
			dbo.fs_getSHA2_512Encode (rtrim(ltrim(@Password)) + @License)
	),
	@Valid int = NULL,
	@Status int = NULL DECLARE @VarifyCode nvarchar(512) = (
		SELECT
			dbo.fs_getSHA2_512Encode (@Account + @PWD + @License)
	)
	--//add a new Object 
INSERT
	Object (
		TYPE,
		CName
	)
VALUES
	(2, @UserName) DECLARE @NewOID int = (@@IDENTITY)
	--//add a new Member
INSERT INTO
	Member (
		MID,
		Account,
		PWD,
		Valid,
		Status,
		VerifyCode,
		EMail,
		Sex,
		Birthday,
		Address,
		Phone
	)
VALUES
	(
		@NewOID,
		@Account,
		@PWD,
		@Valid,
		@Status,
		@VarifyCode,
		@EMail,
		@Sex,
		@Birthday,
		@Address,
		@Phone
	)
	--//add GM
INSERT INTO
	GM (
		GID,
		MID,
		ROLE,
		TYPE,
		Status
	)
VALUES
	(2, @NewOID, 2, 1, 0)
SET
	@NewMID = @NewOID END try BEGIN catch
	--系統拋回訊息用
	DECLARE @ErrorMessage AS VARCHAR(1000) = CHAR(10) + '錯誤代碼：' + CAST(ERROR_NUMBER() AS VARCHAR) + CHAR(10) + '錯誤訊息：' + ERROR_MESSAGE() + CHAR(10) + '錯誤行號：' + CAST(ERROR_LINE() AS VARCHAR) + CHAR(10) + '錯誤程序名稱：' + ISNULL(ERROR_PROCEDURE(), '') DECLARE @ErrorSeverity AS Numeric = ERROR_SEVERITY() DECLARE @ErrorState AS Numeric = ERROR_STATE() RAISERROR (@ErrorMessage, @ErrorSeverity, @ErrorState);


--回傳錯誤資訊
END catch END;


GO
--
IF EXISTS (
	SELECT
		*
	FROM
		sys.objects o
	WHERE
		o.name = N'xp_deleteClass'
		AND
	TYPE IN ('P')
) DROP
PROCEDURE xp_deleteClass;


GO
CREATE PROCEDURE
	xp_deleteClass @CID int AS BEGIN
SET
XACT_ABORT ON --指定當 Transact-SQL 陳述式產生執行階段錯誤時，SQL Server 是否自動回復目前的交易
BEGIN try BEGIN TRANSACTION --下面的過程設定為一整筆交易動作
DELETE CO
WHERE
	CID = @CID
DELETE Permission
WHERE
	CID = @CID
DELETE Inheritance
WHERE
	CCID = @CID
	OR PCID = @CID
DELETE Class
WHERE
	CID = @CID COMMIT TRANSACTION END try BEGIN catch IF XACT_STATE() <> 0 BEGIN ROLLBACK TRANSACTION END
SELECT
	ERROR_NUMBER() AS ErrorNumber,
	ERROR_MESSAGE() AS ErrorMessage END catch
SET
XACT_ABORT OFF END;
