IF
    NOT EXISTS (
        SELECT * FROM INFORMATION_SCHEMA.TABLES
        WHERE
            TABLE_NAME = 'MetaData'
            AND TABLE_SCHEMA = 'dbo'
    )
    BEGIN
        CREATE TABLE Dbo.MetaData (
            MID int NOT NULL,
            Prompt nvarchar(MAX) NULL,
            SourceUpdated datetime NULL,
            LLMResponse nvarchar(MAX) NULL,
            TargetColumnName nvarchar(100) NULL,
            LogicAction nvarchar(MAX) DEFAULT '[]',
            ProcessingHistory nvarchar(MAX) DEFAULT '[]',
            [Status] nvarchar(50) DEFAULT 'unavailable',
            [ApplierStatus] nvarchar(50) DEFAULT 'unavailable',
            CONSTRAINT PK_Meta_MID PRIMARY KEY CLUSTERED (MID ASC),
            CONSTRAINT FK_Meta_MID FOREIGN KEY (MID) REFERENCES [Object] (OID)
        );
    END;


GO

CREATE OR ALTER VIEW Vd_Metadata
AS
SELECT
    MID AS Metadata_Id,
    Prompt AS Prompt,
    SourceUpdated AS Source_Updated,
    LLMResponse AS Llm_Response,
    LogicAction AS Logic_Action,
    ProcessingHistory AS Processing_History,
    TargetColumnName AS Target_Column_Name,
    [Status] AS Status,
    [ApplierStatus] AS Applier_Status
FROM Dbo.MetaData

GO

CREATE OR ALTER PROCEDURE Dbo.Xp_Add_Metadata (
    @dataset_id int,
    @user_id int,
    @prompt nvarchar(MAX),
    @llm_response nvarchar(MAX),
    @logic_action nvarchar(MAX),
    @processing_history nvarchar(MAX),
    @target_column_name nvarchar(100),
    @state int OUTPUT,
    @message nvarchar(4000) OUTPUT,
    @new_id int OUTPUT
)
AS BEGIN TRY

    IF
        NOT EXISTS (
            SELECT 1 FROM Vd_Data_Source
            WHERE Oid = @dataset_id AND Owner_Mid = @user_id
        )
        BEGIN
            SELECT
                @state = 0,
                @message = 'data not exists or user has no permission';
            RETURN
        END

    BEGIN TRAN;

    MERGE INTO [dbo].[MetaData] AS T
    USING (
        VALUES (@dataset_id, @prompt, (
            SELECT LastModifiedDT FROM [Object]
            WHERE OID = @dataset_id
        ), @llm_response, @logic_action, @target_column_name, @processing_history)
    ) AS S (Metadata_Id, Prompt, Source_Updated, LLM_Response, Logic_Action, Target_Column_Name, Processing_History)
        ON S.Metadata_Id = T.MID
    WHEN MATCHED
        THEN
        UPDATE
            SET
                T.Prompt = S.Prompt,
                T.LLMResponse = S.LLM_Response,
                T.LogicAction = S.Logic_Action,
                T.SourceUpdated = S.Source_Updated,
                T.ProcessingHistory = S.Processing_History,
                T.TargetColumnName = S.Target_Column_Name,
                T.[Status] = 'complete'
    WHEN NOT MATCHED BY TARGET
        THEN
        INSERT (MID, Prompt, SourceUpdated, LLMResponse, LogicAction, TargetColumnName, ProcessingHistory, [Status])
        VALUES (
            S.Metadata_Id,
            S.Prompt,
            S.Source_Updated,
            S.LLM_Response,
            S.Logic_Action,
            S.Target_Column_Name,
            S.Processing_History,
            'complete'
        );

    COMMIT TRAN;

    SELECT
        @state = 0,
        @message = 'add metadata successfully'

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


CREATE OR ALTER PROCEDURE Dbo.Xp_Update_Metadata_Status (
    @dataset_id int,
    @user_id int,
    @status nvarchar(50),
    @applier_status nvarchar(50),
    @state int OUTPUT,
    @message nvarchar(4000) OUTPUT,
    @new_id int OUTPUT
)
AS BEGIN TRY

    IF
        NOT EXISTS (
            SELECT 1 FROM Vd_Data_Source
            WHERE Oid = @dataset_id AND Owner_Mid = @user_id
        )
        BEGIN
            SELECT
                @state = 0,
                @message = 'data not exists or user has no permission';
            RETURN
        END

    BEGIN TRAN;

    MERGE INTO [dbo].[MetaData] AS T
    USING (
        VALUES (@dataset_id, @status, @applier_status)
    ) AS S ([metadata_id], [Status], [Applier_Status]) ON S.Metadata_Id = T.MID
    WHEN MATCHED
        THEN
        UPDATE
            SET
                T.[Status] = S.[Status],
                T.ApplierStatus = S.[Applier_Status]
    WHEN NOT MATCHED BY TARGET
        THEN
        INSERT (MID, [Status], [ApplierStatus])
        VALUES (S.[metadata_id], S.[Status], S.[Applier_Status]);

    COMMIT TRAN;

    SELECT
        @state = 0,
        @message = 'update metadata status successfully'

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
