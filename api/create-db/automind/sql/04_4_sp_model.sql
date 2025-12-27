USE [AutoML];


GO
-------------------------------------------------------------------------------
-- SP: add model																											    	  |
-------------------------------------------------------------------------------
-- TODO: select data query
CREATE OR ALTER PROCEDURE
	[dbo].[xp_add_model] @mid int,
	@cid int,
	@data_oid int,
	@engine_oid int,
	@name nvarchar(512),
	@des nvarchar(4000),
	@predict nvarchar(128),
	@tag nvarchar(128),
	@new_id int OUTPUT AS BEGIN try
	--

begin tran;

INSERT INTO
	[dbo].[Object] (
		CName,
		CDes,
		OwnerMID,
		TYPE
	)
VALUES
	(@name, @des, @mid, 112);


SELECT
	@new_id = SCOPE_IDENTITY();


MERGE INTO
	[dbo].[Model] AS Target using (
		VALUES
			(@new_id, @predict, @tag)
	) AS Source (id, predict, tag) ON Target.MID = Source.id
WHEN NOT MATCHED THEN
INSERT
	(MID, Predict, Tag)
VALUES
	(id, predict, tag);


INSERT INTO
	CO (CID, OID)
VALUES
	(@cid, @new_id);


IF NOT EXISTS (
	SELECT
		*
	FROM
		CO
	WHERE
		OID = @data_oid
)
INSERT INTO
	CO (CID, OID)
VALUES
	(@cid, @data_oid);


IF NOT EXISTS (
	SELECT
		*
	FROM
		CO
	WHERE
		OID = @engine_oid
)
INSERT INTO
	CO (CID, OID)
VALUES
	(@cid, @engine_oid);


INSERT INTO
	ORel (OID1, OID2)
VALUES
	(@new_id, @data_oid),
	(@new_id, @engine_oid);


UPDATE Object
SET
	DataByte = 1
WHERE
	OID = @data_oid
	OR OID = @engine_oid;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch;


GO
-------------------------------------------------------------------------------
-- SP: delete model																											    	|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_delete_model] @mid int,
	@project_id int,
	@model_id int as begin try
	--

begin tran;

DECLARE @data_id int, @engine_id int;

SELECT
	@data_id = data_source_id,
	@engine_id = engine_id
FROM
	vd_Model
WHERE
	model_id = @model_id;


IF NOT EXISTS (
	SELECT
		*
	FROM
		Object
	WHERE
		OID = @model_id
		AND OwnerMID = @mid
		AND
	TYPE = 112
) begin;
	throw 50403, 'Forbidden, user dosen''t has permission.', 1;
end;


DELETE FROM ORel
WHERE
	OID1 = @model_id and OID2 = @data_id;


DELETE FROM ORel
WHERE
	OID1 = @model_id and OID2 = @engine_id


DELETE FROM CO
WHERE
	CID = @project_id
	AND (OID = @model_id);


DELETE FROM Model
WHERE
	MID = @model_id;


DELETE FROM Object
WHERE
	OID = @model_id;


IF NOT EXISTS (
	SELECT
		*
	FROM
		vd_Model
	WHERE
		data_source_id = @data_id
		AND project_id = @project_id
)
DELETE FROM CO
WHERE
	CID = @project_id
	AND OID = @data_id;


IF NOT EXISTS (
	SELECT
		*
	FROM
		vd_Model
	WHERE
		engine_id = @engine_id
		AND project_id = @project_id
)
DELETE FROM CO
WHERE
	CID = @project_id
	AND OID = @engine_id;


IF NOT EXISTS (
	SELECT
		*
	FROM
		vd_Model
	WHERE
		engine_id = @engine_id
)
UPDATE Object
SET
	DataByte = 0
WHERE
	OID = @engine_id;


IF NOT EXISTS (
	SELECT
		*
	FROM
		vd_Model
	WHERE
		data_source_id = @data_id
)
UPDATE Object
SET
	DataByte = 0
WHERE
	OID = @data_id;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw;
end catch


GO
-------------------------------------------------------------------------------
-- SP: set model info																										    	|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_set_model_info]
	@model_id int,
	@input nvarchar(max),
	@output nvarchar(max)
	AS BEGIN try
	--
begin tran;

UPDATE [dbo].[Model]
SET
	OutputFeatures = @output,
	InputFeatures = @input
WHERE
	MID = @model_id;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch;

GO
-------------------------------------------------------------------------------
-- SP: update model information																					    	|
-------------------------------------------------------------------------------
CREATE OR ALTER PROCEDURE
	[dbo].[xp_update_model] @model_id int,
	@select_data_query nvarchar(MAX),
	@active bit,
	@status nvarchar(50),
	@score nvarchar(1000),
	@training_time float,
	@update_status nvarchar(50),
	@error nvarchar(MAX),
	@current_training_phase int,
	@total_training_phases int,
	@training_options nvarchar(MAX) AS BEGIN try
	--
begin tran;

MERGE INTO
	Object AS Target using (
		VALUES
			(@model_id, @select_data_query)
	) AS Source (id, query) ON Target.OID = Source.id
WHEN MATCHED THEN
UPDATE SET
	Target.EDes = Source.query;


MERGE INTO
	Model AS Target using (
		VALUES
			(
				@model_id,
				@active,
				@status,
				@score,
				@training_time,
				@update_status,
				@error,
				@current_training_phase,
				@total_training_phases,
				@training_options
			)
	) AS Source (
		id,
		active,
		_status,
		score,
		training_time,
		update_status,
		error,
		current_training_phase,
		total_training_phases,
		training_options
	) ON Target.MID = Source.id
WHEN MATCHED THEN
UPDATE SET
	Target.Active = Source.active,
	Target.Status = Source._status,
	Target.Score = Source.score,
	Target.TrainingTime = Source.training_time,
	Target.UpdateStatus = Source.update_status,
	Target.Error = Source.error,
	Target.CurrentTrainingPhase = Source.current_training_phase,
	Target.TotalTrainingPhases = Source.total_training_phases,
	Target.TrainingOptions = Source.training_options;

commit tran;

end try
begin catch
	if @@TRANCOUNT > 0 rollback tran;
	throw 50000, 'internal server error', 1;
end catch


GO
