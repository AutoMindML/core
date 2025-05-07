USE [AutoML];


GO
-------------------------------------------------------------------------------
-- SP: add Project																											    	|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_add_project] @mid int,
	@name nvarchar(256),
	@des nvarchar(4000),
	@new_id int OUTPUT AS BEGIN try
	--
begin tran;

DECLARE @member_project_cid int = [dbo].[fn_get_member_cid] (@mid, 'project');


DECLARE @project_name nvarchar(256) = concat(
	@name,
	'_',
	cast(
		(
			SELECT
				count(*)
			FROM
				Class C
				LEFT JOIN Inheritance I ON C.CID = I.PCID
			WHERE
				C.nLevel = 3
				AND I.PCID = @member_project_cid
		) AS nvarchar(MAX)
	)
);


EXEC xp_insertClass @member_project_cid,
106,
@project_name,
@des,
@mid,
@new_id OUTPUT;


UPDATE class
SET
	EName = @name
WHERE
	CID = @new_id
	AND OwnerMID = @mid;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch;


GO
-------------------------------------------------------------------------------
-- SP: delete project																											   	|
-------------------------------------------------------------------------------
-- this sp will make sure project doesn't has any object
CREATE OR ALTER PROCEDURE
	[dbo].[xp_delete_project] @cid int,
	@mid int as begin try
	--

begin tran;

IF EXISTS (
		SELECT
			CO.OID
		FROM
			Class C
			INNER JOIN CO ON CO.CID = C.CID
		WHERE
			C.CID = @cid
			AND OwnerMID = @mid
			AND NamePath LIKE 'member/%/project/%'
			AND nLevel = 4
) begin;
	throw 50409, 'Conflict, this project has resouces, please delete them before delete project', 1;
end;


EXEC [dbo].[xp_deleteClass] @cid;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw;
end catch;


GO
