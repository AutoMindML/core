/****** Object:  StoredProcedure [dbo].[xp_delete_data_source]    Script Date: 2025/10/19 下午 12:51:38 ******/
SET ANSI_NULLS ON
GO


SET QUOTED_IDENTIFIER ON
GO

ALTER PROCEDURE [dbo].[xp_delete_data_source]
    @user_id int,
    @dataset_id int,
    @state int OUTPUT,
    @message nvarchar(4000) OUTPUT,
    @new_id int OUTPUT
AS BEGIN TRY
    BEGIN TRAN;

    DECLARE @data_source_cid int = [dbo].[fn_get_member_data_source_cid](@user_id);

    IF
        NOT EXISTS (
            SELECT *
            FROM
                [dbo].[vd_Data_Source]
            WHERE
                Oid = @dataset_id
                AND ((Used_Status = 0) OR (Used_Status IS null))
                AND Owner_Mid = @user_id
        )
        BEGIN
            SELECT
                @state = 1,
                @message = 'dataset not exists or user has no permission';
            COMMIT TRAN;
            RETURN;
        END;

    DELETE FROM CO
    WHERE
        CID = @data_source_cid
        AND OID = @dataset_id;

    IF
        NOT EXISTS (
            SELECT *
            FROM
                [Data_Source]
            WHERE
                DSID = @dataset_id
        )
        BEGIN
            DELETE FROM [DFM]
            WHERE
                DID = @dataset_id;
            DELETE FROM [Object]
            WHERE
                OID = @dataset_id;
        END
    ELSE
        UPDATE [Object]
        SET
            BDel = 1
        WHERE
            OID = @dataset_id;

    SELECT
        @state = 0,
        @message = 'delete dataset successfully';
    COMMIT TRAN;

END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    RAISERROR ('internal server error', 18, 1);
END CATCH;

GO
