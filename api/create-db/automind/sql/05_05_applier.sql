CREATE OR ALTER PROCEDURE Dbo.Xp_Applier_Generate_New_Dataset (
    @name nvarchar(100),
    @des nvarchar(4000),
    @user_id int,
    @md5 varchar(32),
    @origin_dataset_id int,

    @rows int,
    @cols int,
    @col_names nvarchar(MAX),
    @col_types nvarchar(max),
    @size float,
    @size_unit nvarchar(50),
    @quality float,

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

    --IF
    --    NOT EXISTS (
    --        SELECT MD5
    --        FROM
    --            [Data_Source]
    --        WHERE
    --            MD5 = @binary_md5
    --    )
    --    INSERT INTO
    --    [Data_Source] (DSID, MD5, ConnectionData)
    --    VALUES
    --    (@new_id, @binary_md5, null);


    ----
    merge into [dbo].[Data_Source] as T
    using (
        VALUES (
			@new_id, @binary_md5, null, @rows, @cols, @col_names, @size, @size_unit, @quality, @col_types
        )
    ) as S (dsid, md5, connectiondata, [RowCount], colcount, columnnames, size, unit, quality, columntypes)
    on (T.md5 = S.md5)
    when matched
        then
        update
            set
                T.connectiondata = S.connectiondata,
                T.[RowCount] = S.[RowCount],
                T.colcount = S.colcount,
                T.columnnames = S.columnnames,
                T.size = S.size,
                T.unit = S.unit,
                T.quality = S.quality,
                T.columntypes = S.columntypes
    when not matched
        then 
        insert (dsid, md5, connectiondata, [RowCount], colcount, columnnames, size, unit, quality, columntypes)
        values (dsid, md5, connectiondata, [RowCount], colcount, columnnames, size, unit, quality, columntypes);
    ----


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
