CREATE TABLE ObjectExt (
	OID int NOT NULL UNIQUE,
	nInlinks int NULL,
	nOutlinks int NULL,
	bPublished bit NULL DEFAULT (1),
	GroupID int NULL,
	OtherDT datetime NOT NULL DEFAULT (getdate()),
	PRIMARY KEY (OID),
	FOREIGN KEY (OID) REFERENCES Object (OID)
);


CREATE TABLE ClassExt (
	CID int NOT NULL UNIQUE,
	cRank tinyint NULL DEFAULT (0),
	oRank tinyint NULL DEFAULT (0),
	nClickTrue int NOT NULL DEFAULT (0),
	Property smallint NOT NULL DEFAULT (0),
	Code nvarchar(255) NULL,
	PRIMARY KEY (CID),
	FOREIGN KEY (CID) REFERENCES Class (CID)
);


CREATE NONCLUSTERED
INDEX IX_Class_CName ON Class (CName ASC);


GO;
