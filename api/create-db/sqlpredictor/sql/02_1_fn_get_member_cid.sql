USE [AutoML];


GO
-------------------------------------------------------------------------------
-- Get member's data source cid 																			    	  |
-------------------------------------------------------------------------------
CREATE
OR ALTER
FUNCTION [dbo].[fn_get_member_data_source_cid] (@mid int) returns int AS BEGIN DECLARE @cid int;


SELECT
	@cid = CID
FROM
	Class
WHERE
	NamePath = 'member/' + (
		SELECT
			Account
		FROM
			[dbo].[Member]
		WHERE
			MID = @mid
	) + '/data_source';


RETURN @cid END;


GO
-------------------------------------------------------------------------------
-- Get member's cid from given directory name 												    	  |
-------------------------------------------------------------------------------
CREATE
OR ALTER
FUNCTION [dbo].[fn_get_member_cid] (@mid int, @dir nvarchar(256)) returns int AS BEGIN DECLARE @cid int;


SELECT
	@cid = CID
FROM
	Class
WHERE
	NamePath = 'member/' + (
		SELECT
			Account
		FROM
			[dbo].[Member]
		WHERE
			MID = @mid
	) + '/' + @dir;


RETURN @cid END;


GO
