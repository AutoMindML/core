USE [AutoML];


GO
-------------------------------------------------------------------------------
-- SP: add app prediction																						    			|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_add_app_prediction] @mid int,
	@project_id int,
	@model_id int,
	@name nvarchar(512),
	@des nvarchar(4000),
	@new_id int OUTPUT AS BEGIN try
	--
begin tran;

INSERT INTO
	Object (
		TYPE,
		CName,
		CDes,
		OwnerMID,
		nOutlinks
	)
VALUES
	(113, @name, @des, @mid, 1);


SELECT
	@new_id = SCOPE_IDENTITY();


INSERT INTO
	App_Prediction (APID, Status, [Key])
VALUES
	(@new_id, 'active', NEWID());


INSERT INTO
	CO (CID, OID)
VALUES
	(@project_id, @new_id);


INSERT INTO
	ORel (OID1, OID2)
VALUES
	(@new_id, @model_id);

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch;


GO
-------------------------------------------------------------------------------
-- SP: delete app prediction																				    			|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_delete_app_prediction] @mid int,
	@app_id int as begin try
	--
begin tran;
	
IF NOT EXISTS (
		SELECT
			*
		FROM
			Object
		WHERE
			OID = @app_id
			AND OwnerMID = @mid
			AND
		TYPE = 113
) begin
	;throw 50409, 'Forbidden, user does not has permission.', 1;
end;


DELETE FROM ORel
WHERE
	OID1 = @app_id;


DELETE FROM CO
WHERE
	OID = @app_id;


DELETE FROM App_Prediction
WHERE
	APID = @app_id;


DELETE FROM Object
WHERE
	OID = @app_id;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw;
end catch;

go
