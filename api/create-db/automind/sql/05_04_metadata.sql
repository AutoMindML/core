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
            ParsedAction nvarchar(MAX) NULL,
            ParsedHistory nvarchar(MAX) NULL,
            TargetColumnName nvarchar(100) NULL,
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
    ParsedAction AS Parsed_Action,
    ParsedHistory AS Parsed_History,
    TargetColumnName AS Target_Column_Name
FROM Dbo.MetaData

GO

CREATE OR ALTER PROCEDURE Dbo.Xp_Add_Metadata (
    @dataset_id int,
    @user_id int,
    @prompt nvarchar(MAX),
    @llm_response nvarchar(MAX),
    @parsed_action nvarchar(MAX),
    @parsed_history nvarchar(MAX),
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
        ), @llm_response, @parsed_action, @target_column_name, @parsed_history)
    ) AS S (Metadata_Id, Prompt, Source_Updated, LLM_Response, Parsed_Action, Target_Column_Name, Parsed_History)
        ON S.Metadata_Id = T.MID
    WHEN MATCHED
        THEN
        UPDATE
            SET
                T.Prompt = S.Prompt,
                T.LLMResponse = S.LLM_Response,
                T.ParsedAction = S.Parsed_Action,
                T.ParsedHistory = S.Parsed_History,
                T.TargetColumnName = S.Target_Column_Name
    WHEN NOT MATCHED BY TARGET
        THEN
        INSERT (MID, Prompt, SourceUpdated, LLMResponse, ParsedAction, TargetColumnName, ParsedHistory)
        VALUES
            (
                S.Metadata_Id,
                S.Prompt,
                S.Source_Updated,
                S.LLM_Response,
                S.Parsed_Action,
                S.Target_Column_Name,
                S.Parsed_History
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
