USE [AutoML];


GO
-------------------------------------------------------------------------------
-- SP: add data source file																						    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_add_data_source_file] @mid int,
	@name nvarchar(512),
	@des nvarchar(4000),
	@new_id int OUTPUT AS BEGIN TRY

begin tran;

INSERT INTO
	[dbo].[Object] (
		CName,
		CDes,
		TYPE,
		DataByte,
		OwnerMID
	)
VALUES
	(@name, @des, 109, 0, @mid);


SELECT
	@new_id = SCOPE_IDENTITY();


DECLARE @md5 binary(16),
@calc_md5_query nvarchar(MAX),
@calc_md5_params nvarchar(MAX);


SET
	@calc_md5_query = N'select @md5 = hashbytes(''MD5'', (select * from [dbo].[' + CAST(@mid AS nvarchar(MAX)) + N'_' + @name + N'] for json path))';


SET
	@calc_md5_params = N'@md5 binary(16) output';


EXEC sp_executesql @calc_md5_query,
@calc_md5_params,
@md5 = @md5 OUTPUT;


IF NOT EXISTS (
	SELECT
		MD5
	FROM
		Data_Source
	WHERE
		MD5 = @md5
)
INSERT INTO
	Data_Source (DSID, MD5, ConnectionData)
VALUES
	(@new_id, @md5, NULL);


UPDATE Object
SET
	EName = convert(varchar(128), @md5, 2)
WHERE
	OID = @new_id;


DECLARE @clear_query nvarchar(MAX) = N'drop table [dbo].[' + CAST(@mid AS nvarchar(MAX)) + N'_' + @name + N']';


EXEC sp_executesql @clear_query;


DECLARE @data_source_cid int = (
	SELECT
		[dbo].[fn_get_member_data_source_cid] (@mid)
);


INSERT INTO
	CO (CID, OID)
VALUES
	(@data_source_cid, @new_id);

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch;


GO
-------------------------------------------------------------------------------
-- SP: add data source database																				    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_add_data_source_database] @mid int,
	@name nvarchar(512),
	@des nvarchar(4000),
	@engine nvarchar(50),
	@connection_args nvarchar(MAX),
	@new_id int OUTPUT AS BEGIN try

begin tran;

INSERT INTO
	[dbo].[Object] (
		CName,
		CDes,
		TYPE,
		DataByte,
		OwnerMID
	)
VALUES
	(
		@name,
		@des,
		(
			SELECT
				EID
			FROM
				Entity
			WHERE
				EName = 'data:' + @engine
		),
		0,
		@mid
	);


SELECT
	@new_id = SCOPE_IDENTITY();


DECLARE @md5 binary(16) = HASHBYTES('MD5', CONCAT(@engine, @connection_args));


DECLARE @hash_password varchar(128) = convert(
	varchar(128),
	pwdencrypt(
		(
			SELECT
				json_value(@connection_args, '$.password')
		)
	),
	2
);


DECLARE @connection_data nvarchar(MAX) = (
	SELECT
		json_modify(@connection_args, '$.password', @hash_password)
);


IF NOT EXISTS (
	SELECT
		MD5
	FROM
		Data_Source
	WHERE
		MD5 = @md5
)
INSERT INTO
	Data_Source (DSID, MD5, ConnectionData)
VALUES
	(@new_id, @md5, @connection_data);


UPDATE Object
SET
	EName = convert(varchar(128), @md5, 2)
WHERE
	OID = @new_id;


DECLARE @data_source_cid int = (
	SELECT
		[dbo].[fn_get_member_data_source_cid] (2)
);


INSERT INTO
	CO (CID, OID)
VALUES
	(@data_source_cid, @new_id);

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch;


GO
