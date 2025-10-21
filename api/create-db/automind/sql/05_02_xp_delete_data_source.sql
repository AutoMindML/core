/****** Object:  StoredProcedure [dbo].[xp_delete_data_source]    Script Date: 2025/10/19 下午 12:51:38 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

ALTER PROCEDURE [dbo].[xp_delete_data_source] 
	@user_id int,
	@dataset_id int,
	@state int output,
	@message nvarchar(4000) output,
	@new_id int output
as begin try
begin tran;

DECLARE @data_source_cid int = [dbo].[fn_get_member_data_source_cid] (@user_id);

IF NOT EXISTS (
	SELECT
		*
	FROM
		[dbo].[vd_Data_Source]
	WHERE
		oid = @dataset_id
		AND ((used_status = 0) or (used_status is null))
		AND owner_mid = @user_id
) begin
	select
		@state = 1,
		@message = 'dataset not exists or user has no permission';
	commit tran;
	return;
end;

DELETE FROM CO
WHERE
	CID = @data_source_cid
	AND OID = @dataset_id;

IF NOT EXISTS (
	SELECT
		*
	FROM
		[Data_Source]
	WHERE
		DSID = @dataset_id
)
begin
DELETE FROM [DFM]
WHERE
	DID = @dataset_id;
DELETE FROM [Object]
WHERE
	OID = @dataset_id;
end
ELSE
UPDATE [Object]
SET
	bDel = 1
WHERE
	OID = @dataset_id;

select
	@state = 0,
	@message = 'delete dataset successfully';
commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	raiserror ('internal server error', 18, 1);
end catch;

GO


