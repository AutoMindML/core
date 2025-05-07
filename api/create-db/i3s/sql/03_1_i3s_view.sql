USE [$(DBName)];


GO
/****** Object:  View [dbo].[vs_Archive]    Script Date: 2024/12/3 下午 02:15:56 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
CREATE VIEW [dbo].[vs_Archive] AS
SELECT
	o.OID,
	O.[Type],
	o.CName,
	o.CDes,
	a.[FileName],
	a.FileExtension,
	c.Title AS MIMEType,
	o.nClick,
	a.Keywords,
	a.Lang,
	a.Indexable,
	a.IndexInfo,
	o.Since,
	o.LastModifiedDT,
	o.bHided,
	o.bDel
FROM
	Archive a,
	Object o,
	ContentType c
WHERE
	a.AID = o.OID
	AND a.ContentType = c.CTID;


GO
/****** Object:  View [dbo].[vs_Member]    Script Date: 2024/12/3 下午 02:15:56 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
CREATE VIEW [dbo].[vs_Member] AS
SELECT
	o.OID,
	o.[Type],
	o.CName,
	o.CDes,
	o.EName,
	o.EDes,
	m.Account,
	m.PWD,
	m.Valid,
	m.Status,
	m.VerifyCode,
	m.EMail,
	m.Phone,
	m.Address,
	m.Birthday,
	m.Nation,
	m.ClassID,
	m.SendEMailOK,
	m.LastLoginDT,
	m.LoginCount,
	m.LoginErrCount,
	o.Since,
	o.LastModifiedDT,
	o.bHided,
	o.bDel
FROM
	Member m,
	Object o
WHERE
	m.MID = o.OID;


GO
/****** Object:  View [dbo].[vs_ObjectList]    Script Date: 2024/12/3 下午 02:15:56 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
CREATE VIEW [dbo].[vs_ObjectList] AS
SELECT
	co.CID,
	co.Rank,
	o.OID,
	o.Type,
	o.CName,
	o.Since,
	o.LastModifiedDT,
	o.OwnerMID,
	o.nClick,
	o.bHided,
	o.bDel
FROM
	CO,
	Object o
WHERE
	co.OID = o.OID;


GO
/****** Object:  View [dbo].[vs_Post]    Script Date: 2024/12/3 下午 02:15:56 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
CREATE VIEW [dbo].[vs_Post] AS
SELECT
	o.OID,
	o.Type,
	o.CName,
	o.CDes,
	p.Detail,
	p.StartDT,
	p.EndDT,
	o.Since,
	o.LastModifiedDT,
	o.OwnerMID,
	o.bHided,
	o.bDel
FROM
	Post p,
	Object o
WHERE
	p.PID = O.OID;


GO
/****** Object:  View [dbo].[vs_SubClass]    Script Date: 2024/12/3 下午 02:15:56 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER OFF;


GO
CREATE VIEW [dbo].[vs_SubClass] AS
SELECT
	i.PCID AS CID,
	i.Rank,
	c.CID AS CCID,
	c.Type,
	c.CName,
	c.CDes,
	c.Since,
	c.LastModifiedDT,
	c.nObject,
	c.nClick,
	c.Keywords,
	c.OwnerMID,
	c.bHided,
	c.bDel
FROM
	Inheritance i,
	Class c
WHERE
	i.CCID = c.CID;


GO
