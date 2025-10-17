USE [AutoML];


GO
-------------------------------------------------------------------------------
-- Data Source																												    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW vd_Data_Source AS
SELECT
	O.OID AS oid,
	(
		SELECT
			VALUE
		FROM
			STRING_SPLIT (
				(
					SELECT
						EName
					FROM
						Entity
					WHERE
						EID = O.Type
				),
				':',
				1
			)
		ORDER BY
			ordinal ASC
		OFFSET
			1 ROWS
		FETCH NEXT
			1 ROWS ONLY
	) AS source_type,
	O.CName AS name,
	O.CDes AS description,
	O.EName AS md5,
	O.Since AS created_at,
	O.LastModifiedDT AS updated_at,
	O.DataByte AS used_status,
	O.bHided AS is_hided,
	O.bDel AS is_deleted,
	C.CID AS cid,
	C.OwnerMID AS owner_mid
FROM
	Object O
	LEFT JOIN Class C ON C.NamePath LIKE 'member/%/data_source'
WHERE
	(
		SELECT
			EName
		FROM
			Entity
		WHERE
			EID = O.Type
	) LIKE 'data:%'
	AND O.bDel != 1;


GO
-------------------------------------------------------------------------------
-- ML Engine  																												    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vd_ML_Engine] AS
SELECT
	O.OID AS oid,
	(
		SELECT
			VALUE
		FROM
			STRING_SPLIT (
				(
					SELECT
						EName
					FROM
						Entity
					WHERE
						EID = O.Type
				),
				':',
				1
			)
		ORDER BY
			ordinal ASC
		OFFSET
			1 ROWS
		FETCH NEXT
			1 ROWS ONLY
	) AS engine_type,
	O.CName AS name,
	O.CDes AS description,
	O.EName AS md5,
	O.Since AS created_at,
	O.LastModifiedDT AS updated_at,
	O.DataByte AS used_status,
	O.OwnerMID AS owner_mid,
	O.bHided AS is_hided,
	O.bDel AS is_deleted,
	E.Handler AS handler,
	E.ConnectionData AS connection_data
FROM
	[dbo].[Object] O
	LEFT JOIN [dbo].[ML_Engine] E ON O.OID = E.MLEID
WHERE
	O.Type = 111;


GO
-------------------------------------------------------------------------------
-- Project    																												    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vd_Project] AS
SELECT
	CID AS cid,
	EName AS name,
	CDes AS description,
	Since AS created_at,
	LastModifiedDT AS updated_at,
	OwnerMID AS owner_mid
FROM
	Class
WHERE
	NamePath LIKE 'member/%/project/%'
	AND bHided = 0
	AND bDel = 0;


GO
-------------------------------------------------------------------------------
-- Model       																												    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vd_Model] AS
SELECT
	CO.CID AS project_id,
	O.OID AS model_id,
	toApp.OID1 AS app_id,
	(
		SELECT
			[Key]
		FROM
			App_Prediction
		WHERE
			APID = toApp.OID1
	) AS api_key,
	O.CName AS name,
	O.CDes AS description,
	M.OutputFeatures AS output_features,
	M.InputFeatures AS input_features,
	O.EDes AS select_data_query,
	O.Since AS created_at,
	O.LastModifiedDT AS updated_at,
	O.OwnerMID AS owner_mid,
	M.Active AS active,
	M.Version AS version,
	M.Status AS status,
	M.Accuracy AS accuracy,
	M.Predict AS predict,
	M.LearningType AS learning_type,
	M.TaskType AS task_type,
	M.TrainingTime AS training_time,
	M.UpdateStatus AS update_status,
	M.Error AS error,
	M.TrainingOptions AS training_options,
	M.CurrentTrainingPhase AS current_training_phase,
	M.TotalTrainingPhases AS total_training_phases,
	M.Tag AS tag,
	toData.OID2 AS data_source_id,
	(
		SELECT
			EName
		FROM
			[Object]
		WHERE
			OID = toData.OID2
	) AS data_source_md5,
	(
		select dbo.fn_get_data_source_type ((
			SELECT
				[Type]
			FROM
				[Object]
			WHERE
				OID = toData.OID2
		))
	) AS data_source_type,
	toEngine.OID2 AS engine_id,
	(
		SELECT
			EName
		FROM
			Object
		WHERE
			OID = toEngine.OID2
	) AS engine_md5
FROM
	Object O
	LEFT JOIN [dbo].[Model] M ON M.MID = O.OID
	LEFT JOIN CO ON CO.OID = O.OID
	LEFT JOIN ORel toData ON toData.OID1 = O.OID
	AND EXISTS (
		SELECT
			*
		FROM
			vd_Data_Source
		WHERE
			oid = toData.OID2
	)
	LEFT JOIN ORel toEngine ON toEngine.OID1 = O.OID
	AND EXISTS (
		SELECT
		TYPE
		FROM
			Object
		WHERE
			OID = toEngine.OID2
			AND
		TYPE = 111
	)
	LEFT JOIN ORel toApp ON toApp.OID2 = O.OID
	AND EXISTS (
		SELECT
		TYPE
		FROM
			Object
		WHERE
			OID = toApp.OID1
			AND
		TYPE = 113
	)
WHERE
	O.Type = 112;


GO
-------------------------------------------------------------------------------
-- App Prediction	  																									    	  |
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vd_App_Prediction] AS
SELECT
	C.CID AS project_id,
	O.OID AS app_id,
	(
		SELECT
			EName
		FROM
			Entity
		WHERE
			EID = O.Type
	) AS app_type,
	O.CName AS name,
	O.CDes AS description,
	O.Since AS created_at,
	O.LastModifiedDT AS updated_at,
	O.OwnerMID AS owner_mid,
	O.nOutlinks AS active_models,
	A.[Key] AS api_key,
	A.Status AS app_status
FROM
	[dbo].[Object] O
	LEFT JOIN [dbo].[CO] CO ON CO.OID = O.OID
	LEFT JOIN [dbo].[App_Prediction] A ON O.OID = A.APID
	LEFT JOIN [dbo].[Class] C ON CO.CID = C.CID
WHERE
	O.TYPE IN (113);


GO
-------------------------------------------------------------------------------
-- Valid API Keys		      																					    			|
-------------------------------------------------------------------------------
CREATE OR ALTER VIEW [dbo].[vd_valid_api_keys] AS
SELECT
	api_key
FROM
	[dbo].[vd_App_Prediction];


GO
