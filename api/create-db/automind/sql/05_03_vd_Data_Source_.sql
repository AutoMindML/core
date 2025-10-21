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
	[Object] O
	, Class C
	, CO
WHERE
	(
		SELECT
			EName
		FROM
			Entity
		WHERE
			EID = O.Type
	) LIKE 'data:%'
	AND O.bDel != 1
	and C.NamePath LIKE 'member/%/data_source'
	and CO.CID = C.CID and O.OID = CO.OID

GO