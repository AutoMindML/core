IF
    NOT EXISTS (
        SELECT * FROM INFORMATION_SCHEMA.TABLES
        WHERE
            TABLE_NAME = 'App_Prediction'
            AND TABLE_SCHEMA = 'dbo'
    )
    BEGIN
        CREATE TABLE [dbo].[App_Prediction] (
            [APID] [int] NOT NULL,
            [Key] [nvarchar](64) NOT NULL,
            [Status] [nvarchar](50) NULL,
            [DeploymentId] varchar(255),
            CONSTRAINT [PK_App_Prediction] PRIMARY KEY CLUSTERED ([APID] ASC)
            WITH
            (
                PAD_INDEX = OFF,
                STATISTICS_NORECOMPUTE = OFF,
                IGNORE_DUP_KEY = OFF,
                ALLOW_ROW_LOCKS = ON,
                ALLOW_PAGE_LOCKS = ON,
                OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
            ) ON [PRIMARY],
            CONSTRAINT [UQ_App_Prediction_Key] UNIQUE NONCLUSTERED ([Key] ASC)
            WITH
            (
                PAD_INDEX = OFF,
                STATISTICS_NORECOMPUTE = OFF,
                IGNORE_DUP_KEY = OFF,
                ALLOW_ROW_LOCKS = ON,
                ALLOW_PAGE_LOCKS = ON,
                OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
            ) ON [PRIMARY]
        ) ON [PRIMARY];
    END;


GO


CREATE OR ALTER VIEW [dbo].[vd_App_Prediction] AS
SELECT
    C.CID AS Project_Id,
    O.OID AS App_Id,
    (
        SELECT EName
        FROM
            Entity
        WHERE
            EID = O.Type
    ) AS App_Type,
    O.CName AS Name,
    O.CDes AS Description,
    O.Since AS Created_At,
    O.LastModifiedDT AS Updated_At,
    O.OwnerMID AS Owner_Mid,
    O.NOutlinks AS Active_Models,
    A.[Key] AS Api_Key,
    A.[Status] AS App_Status,
    A.DeploymentId AS Deployment_Id
FROM
    [dbo].[Object] O
LEFT JOIN [dbo].[CO] CO ON CO.OID = O.OID
LEFT JOIN [dbo].[App_Prediction] A ON O.OID = A.APID
LEFT JOIN [dbo].[Class] C ON CO.CID = C.CID
WHERE
    O.TYPE IN (113);


GO

CREATE OR ALTER PROCEDURE
[dbo].[xp_add_app_prediction]
    @mid int,
    @project_id int,
    @model_id int,
    @name nvarchar(512),
    @des nvarchar(4000),
    @new_id int OUTPUT AS BEGIN TRY
    BEGIN TRAN;

    INSERT INTO
    Object (
        TYPE,
        CName,
        CDes,
        OwnerMID,
        NOutlinks
    )
    VALUES
    (113, @name, @des, @mid, 1);

    SELECT @new_id = scope_identity()
    ;

    INSERT INTO
    App_Prediction (APID, Status, [Key], [DeploymentId])
    VALUES
    (@new_id, 'active', newid(), newid());

    INSERT INTO
    CO (CID, OID)
    VALUES
    (@project_id, @new_id);

    INSERT INTO
    ORel (OID1, OID2)
    VALUES
    (@new_id, @model_id);

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
