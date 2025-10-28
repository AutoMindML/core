CREATE OR ALTER PROCEDURE Dbo.Xp_Applier_Generate_New_Dataset (
    @name nvarchar(100),
    @des nvarchar(4000),
    @user_id int,
    @md5 varchar(32),
    @origin_dataset_id int,
    @state int OUTPUT,
    @message nvarchar(4000) OUTPUT,
    @new_id int OUTPUT
)
AS BEGIN TRY
    BEGIN TRAN;

    DECLARE @binary_md5 binary(16) = CONVERT(binary(16), @md5, 2);
    SET @md5 = CONVERT(varchar(32), @binary_md5, 2);

    INSERT INTO Dbo.[Object] (CName, CDes, [Type], OwnerMID, DataByte, EName, EDes)
    VALUES (@name, @des, 109, @user_id, 0, @md5, CAST(@origin_dataset_id AS nvarchar(MAX)));

    SELECT @new_id = scope_identity();

    DECLARE @data_source_cid int = (
        SELECT dbo.fn_get_member_data_source_cid(@user_id)
    );

    INSERT INTO CO (CID, OID)
    VALUES (@data_source_cid, @new_id);

    IF
        NOT EXISTS (
            SELECT MD5
            FROM
                [Data_Source]
            WHERE
                MD5 = @binary_md5
        )
        INSERT INTO
        [Data_Source] (DSID, MD5, ConnectionData)
        VALUES
        (@new_id, @binary_md5, null);

    SELECT
        @state = 0,
        @message = 'applier generate new dataset successfully';

    COMMIT TRAN;
END TRY
BEGIN CATCH;
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE
        @error_message nvarchar(4000) = error_message(),
        @error_severity int = error_severity(),
        @error_state int = error_state();
    RAISERROR (@error_message, @error_severity, @error_state);
END CATCH;

GO
