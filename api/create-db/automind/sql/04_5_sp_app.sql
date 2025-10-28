USE [AutoML];


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
