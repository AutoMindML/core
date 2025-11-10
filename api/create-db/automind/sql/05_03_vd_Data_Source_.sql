--ALTER TABLE [dbo].[Data_Source]
--ADD
--[RowCount] int DEFAULT 0,
--[ColCount] int DEFAULT 0,
--[ColumnNames] nvarchar(MAX) NULL,
--[ColumnTypes] nvarchar(max) null,
--[Size] float DEFAULT 0.0,
--[Unit] nvarchar(50) DEFAULT 'MB',
--[Quality] float NULL;
--GO

CREATE OR ALTER VIEW vd_data_source AS
SELECT
    o.oid AS oid,
    (
        SELECT value
        FROM
            string_split(
                (
                    SELECT ename
                    FROM
                        entity
                    WHERE
                        eid = o.type
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
    o.cname AS name,
    o.cdes AS description,
    o.ename AS md5,
    o.since AS created_at,
    o.lastmodifieddt AS updated_at,
    o.databyte AS used_status,
    o.bhided AS is_hided,
    o.bdel AS is_deleted,
    c.cid AS cid,
    c.ownermid AS owner_mid,
    d.[RowCount] AS [rows],
    d.[ColCount] AS [cols],
    d.[ColumnNames] AS [col_names],
    d.[ColumnTypes] AS [col_types],
    d.[Size] AS [size],
    d.[Unit] AS [size_unit],
    d.[Quality] AS [quality]
FROM
    [Class] c, [CO], [Object] o
LEFT JOIN [Data_Source] d ON convert(varchar(32), d.md5, 2) = o.ename
WHERE
    (
        SELECT ename
        FROM
            entity
        WHERE
            eid = o.type
    ) LIKE 'data:%'
    AND o.bdel != 1
    AND c.namepath LIKE 'member/%/data_source'
    AND co.cid = c.cid AND o.oid = co.oid

GO
