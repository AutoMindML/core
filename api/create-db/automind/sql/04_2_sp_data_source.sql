CREATE OR ALTER PROCEDURE
[dbo].[xp_add_data_source_file]
    @user_id int,
    @name nvarchar(512),
    @des nvarchar(4000),
    @md5 varchar(32),
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

AS BEGIN TRY

    BEGIN TRAN;

    INSERT INTO
    [dbo].[Object] (
        cname,
        cdes,
        [type],
        databyte,
        ownermid
    )
    VALUES
    (@name, @des, 109, 0, @user_id);

    SELECT @new_id = scope_identity();

    DECLARE @binary_md5 binary(16) = CONVERT(binary(16), @md5, 2);
    SET @md5 = CONVERT(varchar(32), @binary_md5, 2);


    merge into [dbo].[Data_Source] as T
    using (
        VALUES (
            @new_id, @binary_md5, null, @rows, @cols, @col_names, @size, @size_unit, @quality, @col_types
        )
    ) as S (dsid, md5, connectiondata, [RowCount], colcount, columnnames, size, unit, quality, columntypes)
    on T.md5 = S.md5
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

    UPDATE object
    SET
        ename = @md5
    WHERE
        oid = @new_id;

    DECLARE @data_source_cid int = (
        SELECT [dbo].[fn_get_member_data_source_cid](@user_id)
    );

    INSERT INTO
    co (cid, oid)
    VALUES
    (@data_source_cid, @new_id);

    SELECT
        @state = 0,
        @message = 'add dataset successfully';

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

CREATE OR ALTER PROCEDURE
[dbo].[xp_add_data_source_database]
    @user_id int,
    @name nvarchar(512),
    @des nvarchar(4000),
    @engine nvarchar(50),
    @connection_args nvarchar(MAX),
    @new_id int OUTPUT AS BEGIN TRY

    BEGIN TRAN;

    INSERT INTO
    [dbo].[Object] (
        cname,
        cdes,
        type,
        databyte,
        ownermid
    )
    VALUES
    (
        @name,
        @des,
        (
            SELECT eid
            FROM
                entity
            WHERE
                ename = 'data:' + @engine
        ),
        0,
        @user_id
    );

    SELECT @new_id = scope_identity()
    ;

    DECLARE @md5 binary(16) = hashbytes('MD5', concat(@engine, @connection_args));

    DECLARE @hash_password varchar(128) = CONVERT(
        varchar(128),
        pwdencrypt(
            (
                SELECT json_value(@connection_args, '$.password')
            )
        ),
        2
    );

    DECLARE @connection_data nvarchar(MAX) = (
        SELECT json_modify(@connection_args, '$.password', @hash_password)
    );

    IF
        NOT EXISTS (
            SELECT md5
            FROM
                data_source
            WHERE
                md5 = @md5
        )
        INSERT INTO
        data_source (dsid, md5, connectiondata)
        VALUES
        (@new_id, @md5, @connection_data);

    UPDATE object
    SET
        ename = CONVERT(varchar(128), @md5, 2)
    WHERE
        oid = @new_id;

    DECLARE @data_source_cid int = (
        SELECT [dbo].[fn_get_member_data_source_cid](2)
    );

    INSERT INTO
    co (cid, oid)
    VALUES
    (@data_source_cid, @new_id);

    COMMIT TRAN;

END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    THROW 50000, 'internal server error', 1;
END CATCH;


GO
