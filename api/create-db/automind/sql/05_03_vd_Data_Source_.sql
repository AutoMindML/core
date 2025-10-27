CREATE OR ALTER VIEW Vd_Data_Source AS
SELECT
    O.OID AS Oid,
    (
        SELECT VALUE
        FROM
            string_split(
                (
                    SELECT EName
                    FROM
                        Entity
                    WHERE
                        EID = O.Type
                ),
                ':',
                1
            )
        ORDER BY
            Ordinal ASC
            OFFSET
            1 ROWS
            FETCH NEXT
            1 ROWS ONLY
    ) AS Source_Type,
    O.CName AS Name,
    O.CDes AS Description,
    O.EName AS Md5,
    O.Since AS Created_At,
    O.LastModifiedDT AS Updated_At,
    O.DataByte AS Used_Status,
    O.BHided AS Is_Hided,
    O.BDel AS Is_Deleted,
    C.CID AS Cid,
    C.OwnerMID AS Owner_Mid
FROM
    [Object] O, [Class] C, [CO]
WHERE
    (
        SELECT EName
        FROM
            Entity
        WHERE
            EID = O.Type
    ) LIKE 'data:%'
    AND O.BDel != 1
    AND C.NamePath LIKE 'member/%/data_source'
    AND CO.CID = C.CID AND O.OID = CO.OID

GO
