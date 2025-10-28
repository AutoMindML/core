USE [$(DBName)];


GO
/****** Object:  Table [dbo].[App_Prediction]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO

/****** Object:  Table [dbo].[App_Schedule]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[App_Schedule] (
	[ASID] [int] NOT NULL,
	[Key] [nvarchar] (64) NOT NULL,
	[ScheduleStr] [nvarchar] (MAX) NULL,
	[Query] [nvarchar] (MAX) NULL,
	[IfQuery] [nvarchar] (MAX) NULL,
	[Variables] [nvarchar] (MAX) NULL,
	CONSTRAINT [PK_App_Schedule] PRIMARY KEY CLUSTERED ([ASID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_App_Schedule_Key] UNIQUE NONCLUSTERED ([Key] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY];


GO
/****** Object:  Table [dbo].[Archive]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Archive] (
	[AID] [int] NOT NULL,
	[FileName] [nvarchar] (256) NOT NULL,
	[FileExtension] [nvarchar] (100) NULL,
	[Keywords] [nvarchar] (512) NOT NULL,
	[Lang] [tinyint] NULL,
	[Indexable] [bit] NULL,
	[IndexInfo] [nvarchar] (255) NULL,
	[ContentLen] [int] NULL,
	[MD5] [binary] (16) NULL,
	[ContentType] [smallint] NULL,
	[UUID] [nvarchar] (100) NULL,
	CONSTRAINT [PK_Archive_AID] PRIMARY KEY CLUSTERED ([AID] ASC)
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


GO
/****** Object:  Table [dbo].[Class]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Class] (
	[CID] [int] IDENTITY(1, 1) NOT NULL,
	[Type] [smallint] NULL,
	[CName] [nvarchar] (256) NOT NULL,
	[CDes] [nvarchar] (4000) NULL,
	[EName] [nvarchar] (256) NULL,
	[EDes] [nvarchar] (4000) NULL,
	[IDPath] [varchar] (900) NULL,
	[NamePath] [nvarchar] (450) NULL,
	[Since] [datetime] NOT NULL,
	[LastModifiedDT] [datetime] NOT NULL,
	[nObject] [int] NOT NULL,
	[cRank] [tinyint] NULL,
	[oRank] [tinyint] NULL,
	[nLevel] [tinyint] NULL,
	[nClick] [int] NOT NULL,
	[Keywords] [nvarchar] (512) NULL,
	[OwnerMID] [int] NULL,
	[bHided] [bit] NULL,
	[bDel] [bit] NULL,
	CONSTRAINT [PK_Class_CID] PRIMARY KEY CLUSTERED ([CID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_Class_IDPath] UNIQUE NONCLUSTERED ([IDPath] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_Class_NamePath] UNIQUE NONCLUSTERED ([NamePath] ASC)
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


GO
/****** Object:  Table [dbo].[CO]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[CO] (
	[CID] [int] NOT NULL,
	[OID] [int] NOT NULL,
	[Rank] [smallint] NULL,
	[MG] [tinyint] NULL,
	[Des] [nvarchar] (900) NULL,
	CONSTRAINT [PK_CO] PRIMARY KEY CLUSTERED ([CID] ASC, [OID] ASC)
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


GO
/****** Object:  Table [dbo].[ContentType]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[ContentType] (
	[CTID] [smallint] IDENTITY(1, 1) NOT NULL,
	[Title] [varchar] (100) NOT NULL,
	[Des] [nvarchar] (255) NULL,
	CONSTRAINT [PK_ContentType_CTID] PRIMARY KEY CLUSTERED ([CTID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_ContentType_Title] UNIQUE NONCLUSTERED ([Title] ASC)
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


GO
/****** Object:  Table [dbo].[Data_Source]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Data_Source] (
	[DSID] [int] NOT NULL,
	[MD5] [binary] (16) NOT NULL,
	[ConnectionData] [nvarchar] (MAX) NULL,
	CONSTRAINT [PK_DataSouce] PRIMARY KEY CLUSTERED ([DSID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_Data_Source_MD5] UNIQUE NONCLUSTERED ([MD5] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY];


GO
/****** Object:  Table [dbo].[Entity]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Entity] (
	[EID] [smallint] IDENTITY(1, 1) NOT NULL,
	[CName] [nvarchar] (50) NOT NULL,
	[EName] [nvarchar] (50) NOT NULL,
	[bORel] [bit] NOT NULL,
	[bHided] [bit] NULL,
	[bDel] [bit] NULL,
	CONSTRAINT [PK_Entity] PRIMARY KEY CLUSTERED ([EID] ASC)
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


GO
/****** Object:  Table [dbo].[EntityM2DC]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[EntityM2DC] (
	[EID] [smallint] NOT NULL,
	[Field] [nvarchar] (64) NOT NULL,
	[Caption] [nvarchar] (128) NULL,
	[JSONField] [nvarchar] (64) NULL,
	[DCField] [tinyint] NULL,
	[SNo] [tinyint] NULL,
	CONSTRAINT [PK_EntityM2DC] PRIMARY KEY CLUSTERED ([EID] ASC, [Field] ASC)
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


GO
/****** Object:  Table [dbo].[GM]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[GM] (
	[GID] [int] NOT NULL,
	[MID] [int] NOT NULL,
	[Role] [tinyint] NOT NULL,
	[Type] [bit] NULL,
	[Status] [bit] NULL,
	CONSTRAINT [PK_GM] PRIMARY KEY CLUSTERED ([GID] ASC, [MID] ASC)
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


GO
/****** Object:  Table [dbo].[Groups]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Groups] (
	[GID] [int] IDENTITY(1, 1) NOT NULL,
	[GName] [nvarchar] (50) NOT NULL,
	[GDes] [nvarchar] (1024) NULL,
	[Status] [tinyint] NULL,
	[Since] [datetime] NOT NULL,
	[Type] [tinyint] NULL,
	[bHided] [bit] NOT NULL,
	[bDel] [bit] NOT NULL,
	CONSTRAINT [PK_Groups] PRIMARY KEY CLUSTERED ([GID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_Groups] UNIQUE NONCLUSTERED ([GName] ASC)
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


GO
/****** Object:  Table [dbo].[Inheritance]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Inheritance] (
	[PCID] [int] NOT NULL,
	[CCID] [int] NOT NULL,
	[Rank] [smallint] NULL,
	[MG] [tinyint] NULL,
	CONSTRAINT [PK_Inheritance] PRIMARY KEY CLUSTERED ([PCID] ASC, [CCID] ASC)
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


GO
/****** Object:  Table [dbo].[LogDir]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[LogDir] (
	[SID] [int] NOT NULL,
	[CID] [int] NOT NULL,
	[Operation] [bit] NOT NULL,
	[Sort] [tinyint] NULL,
	[VisitDate] [datetime] NOT NULL,
	CONSTRAINT [PK_LogDir] PRIMARY KEY CLUSTERED ([SID] ASC, [VisitDate] ASC)
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


GO
/****** Object:  Table [dbo].[LogError]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[LogError] (
	[SID] [int] NOT NULL,
	[ErrCode] [int] NULL,
	[ErrMsg] [nvarchar] (900) NULL,
	[ThrowDate] [datetime] NOT NULL,
	CONSTRAINT [PK_LogError] PRIMARY KEY CLUSTERED ([SID] ASC, [ThrowDate] ASC)
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


GO
/****** Object:  Table [dbo].[LogMan]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[LogMan] (
	[SID] [int] NOT NULL,
	[TargetType] [bit] NULL,
	[TargetID] [int] NULL,
	[Operation] [bit] NOT NULL,
	[Sort] [tinyint] NULL,
	[IndexCount] [int] NULL,
	[VisitDate] [datetime] NOT NULL,
	CONSTRAINT [PK_LogMan] PRIMARY KEY CLUSTERED ([SID] ASC, [VisitDate] ASC)
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


GO
/****** Object:  Table [dbo].[LogManTx]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[LogManTx] (
	[SID] [int] NOT NULL,
	[Method] [nvarchar] (512) NULL,
	[DataOperation] [bit] NULL,
	[PostString] [nvarchar] (4000) NULL,
	[VisitDate] [datetime] NOT NULL,
	CONSTRAINT [PK_LogManTx] PRIMARY KEY CLUSTERED ([SID] ASC, [VisitDate] ASC)
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


GO
/****** Object:  Table [dbo].[LogObject]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[LogObject] (
	[SID] [int] NOT NULL,
	[OID] [int] NOT NULL,
	[Operation] [bit] NOT NULL,
	[IndexCount] [int] NULL,
	[VisitDate] [datetime] NOT NULL,
	CONSTRAINT [PK_LogObject] PRIMARY KEY CLUSTERED ([SID] ASC, [VisitDate] ASC)
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


GO
/****** Object:  Table [dbo].[LogSearch]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[LogSearch] (
	[SID] [int] NOT NULL,
	[QueryString] [nvarchar] (512) NOT NULL,
	[SearchDate] [datetime] NOT NULL
) ON [PRIMARY];


GO
/****** Object:  Table [dbo].[Member]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Member] (
	[MID] [int] NOT NULL,
	[Account] [varchar] (100) NOT NULL,
	[PWD] [varchar] (300) NULL,
	[Valid] [bit] NULL,
	[EMail] [nvarchar] (100) NOT NULL,
	[Status] [tinyint] NULL,
	[LoginCount] [int] NOT NULL,
	[LastLoginDT] [datetime] NULL,
	[LoginErrCount] [tinyint] NOT NULL,
	[VerifyCode] [varchar] (300) NULL,
	[ClassID] [int] NULL,
	[Sex] [bit] NULL,
	[Birthday] [smalldatetime] NULL,
	[Nation] [smallint] NULL,
	[Address] [nvarchar] (200) NULL,
	[Phone] [nvarchar] (25) NULL,
	[SendEMailOK] [bit] NULL,
	[emailVerified] [datetime2] (7) NULL,
	[Image] [nvarchar] (1000) NULL,
	CONSTRAINT [PK_Member] PRIMARY KEY CLUSTERED ([MID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		UNIQUE NONCLUSTERED ([Account] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_Member_Account] UNIQUE NONCLUSTERED ([Account] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_Member_EMail] UNIQUE NONCLUSTERED ([EMail] ASC)
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


GO
/****** Object:  Table [dbo].[ML_Engine]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[ML_Engine] (
	[MLEID] [int] NOT NULL,
	[MD5] [binary] (16) NOT NULL,
	[Handler] [nvarchar] (50) NULL,
	[ConnectionData] [nvarchar] (MAX) NULL,
	CONSTRAINT [PK_ML_Engine] PRIMARY KEY CLUSTERED ([MLEID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_ML_Engine_MD5] UNIQUE NONCLUSTERED ([MD5] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY];


GO
/****** Object:  Table [dbo].[Model]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Model] (
	[MID] [int] NOT NULL,
	[Active] [bit] NULL,
	[Version] [int] NULL,
	[Status] [nvarchar] (50) NULL,
	[Accuracy] [float] NULL,
	[Predict] [nvarchar] (50) NULL,
	[LearningType] [nvarchar] (100) NULL,
	[TaskType] [nvarchar] (100) NULL,
	[TrainingTime] [float] NULL,
	[UpdateStatus] [nvarchar] (50) NULL,
	[Error] [nvarchar] (MAX) NULL,
	[TrainingOptions] [nvarchar] (MAX) NULL,
	[CurrentTrainingPhase] [nvarchar] (MAX) NULL,
	[TotalTrainingPhases] [nvarchar] (MAX) NULL,
	[Tag] [nvarchar] (100) NULL,
	[InputFeatures] [nvarchar] (MAX) NULL,
	[OutputFeatures] [nvarchar] (MAX) NULL,
	CONSTRAINT [PK_Model] PRIMARY KEY CLUSTERED ([MID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY];


GO
/****** Object:  Table [dbo].[MSession]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[MSession] (
	[SID] [int] IDENTITY(1, 1) NOT NULL,
	[MID] [int] NOT NULL,
	[IP] [varchar] (16) NOT NULL,
	[UserAgent] [int] NULL,
	[PassportCode] [nvarchar] (512) NOT NULL,
	[Since] [datetime] NOT NULL,
	[LastModifiedDT] [datetime] NULL,
	[ExpiredDT] [datetime] NOT NULL,
	CONSTRAINT [PK_MSession] PRIMARY KEY CLUSTERED ([SID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_MSession_PassportCode] UNIQUE NONCLUSTERED ([PassportCode] ASC)
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


GO
/****** Object:  Table [dbo].[Nation]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Nation] (
	[NID] [smallint] IDENTITY(1, 1) NOT NULL,
	[CName] [nvarchar] (50) NOT NULL,
	[EName] [varchar] (50) NOT NULL,
	[CountryCode] [varchar] (25) NULL,
	[ISOCode2] [varchar] (2) NOT NULL,
	[ISOCode3] [varchar] (3) NOT NULL,
	CONSTRAINT [PK_Nation] PRIMARY KEY CLUSTERED ([NID] ASC)
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


GO
/****** Object:  Table [dbo].[OAuthAccount]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[OAuthAccount] (
	[AID] [int] IDENTITY(1, 1) NOT NULL,
	[MID] [int] NOT NULL,
	[type] [nvarchar] (64) NOT NULL,
	[provider] [nvarchar] (128) NOT NULL,
	[providerAccountId] [nvarchar] (128) NOT NULL,
	[refresh_token] [text] NULL,
	[access_token] [text] NULL,
	[expires_at] [int] NULL,
	[token_type] [nvarchar] (1000) NULL,
	[scope] [nvarchar] (1000) NULL,
	[id_token] [text] NULL,
	[session_state] [nvarchar] (1000) NULL,
	CONSTRAINT [PK_OAuthAccount] PRIMARY KEY CLUSTERED ([AID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [OAuthAccount_provider_providerAccountId_key] UNIQUE NONCLUSTERED ([provider] ASC, [providerAccountId] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY];


GO
/****** Object:  Table [dbo].[Object]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Object] (
	[OID] [int] IDENTITY(1, 1) NOT NULL,
	[Type] [smallint] NOT NULL,
	[CName] [nvarchar] (512) NULL,
	[CDes] [nvarchar] (4000) NULL,
	[EName] [nvarchar] (512) NULL,
	[EDes] [nvarchar] (4000) NULL,
	[Since] [datetime] NOT NULL,
	[LastModifiedDT] [datetime] NOT NULL,
	[OtherDT] [datetime] NULL,
	[DataByte] [binary] (1) NULL,
	[OwnerMID] [int] NULL,
	[nClick] [int] NOT NULL,
	[nOutlinks] [int] NULL,
	[nInlinks] [int] NULL,
	[bHided] [bit] NULL,
	[bDel] [bit] NULL,
	CONSTRAINT [PK_Object_OID] PRIMARY KEY CLUSTERED ([OID] ASC)
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


GO
/****** Object:  Table [dbo].[ORel]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[ORel] (
	[OID1] [int] NOT NULL,
	[OID2] [int] NOT NULL,
	[Rank] [int] NULL,
	[Des] [nvarchar] (900) NULL,
	CONSTRAINT [PK_ORel] PRIMARY KEY CLUSTERED ([OID1] ASC, [OID2] ASC)
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


GO
/****** Object:  Table [dbo].[Permission]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Permission] (
	[CID] [int] NOT NULL,
	[RoleType] [bit] NOT NULL,
	[RoleID] [int] NOT NULL,
	[PermissionBits] [tinyint] NOT NULL,
	CONSTRAINT [UQ_Permission] UNIQUE NONCLUSTERED ([CID] ASC, [RoleType] ASC, [RoleID] ASC)
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


GO
/****** Object:  Table [dbo].[Post]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[Post] (
	[PID] [int] NOT NULL,
	[Detail] [nvarchar] (MAX) NULL,
	[StartDT] [datetime] NULL,
	[EndDT] [datetime] NULL,
	CONSTRAINT [PK_Post] PRIMARY KEY CLUSTERED ([PID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY];


GO
/****** Object:  Table [dbo].[StatusCode]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[StatusCode] (
	[Status] [int] NOT NULL,
	[Msg] [nvarchar] (64) NULL,
	[CDes] [nvarchar] (800) NULL,
	CONSTRAINT [PK_Status] PRIMARY KEY CLUSTERED ([Status] ASC)
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


GO
/****** Object:  Table [dbo].[SystemConfig]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[SystemConfig] (
	[Name] [nvarchar] (256) NOT NULL,
	[Des] [nvarchar] (4000) NULL,
	[Since] [datetime] NULL,
	[LastModifiedDT] [datetime] NULL,
	[bDel] [bit] NULL,
	CONSTRAINT [PK_SystemConfig] PRIMARY KEY CLUSTERED ([Name] ASC)
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


GO
/****** Object:  Table [dbo].[URL]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[URL] (
	[UID] [int] NOT NULL,
	[Scheme] [smallint] NOT NULL,
	[HostName] [varchar] (900) NOT NULL,
	[Path] [nvarchar] (900) NOT NULL,
	[Title] [nvarchar] (255) NULL,
	[Des] [nvarchar] (4000) NULL,
	[Lang] [tinyint] NULL,
	[ContentLen] [int] NULL,
	[Keywords] [nvarchar] (255) NULL,
	[Indexable] [bit] NULL,
	[IndexInfo] [nvarchar] (255) NULL,
	[SID] [int] NULL,
	[MD5URL] [binary] (16) NOT NULL,
	[MD5] [binary] (16) NULL,
	[ContentType] [smallint] NULL,
	[Weight] [tinyint] NULL,
	[Crawl] [tinyint] NULL,
	[ModifiedFreq] [int] NULL,
	[OKFreq] [int] NULL,
	CONSTRAINT [PK_URL_UID] PRIMARY KEY CLUSTERED ([UID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_URL_MD5] UNIQUE NONCLUSTERED ([MD5] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_URL_MD5URL] UNIQUE NONCLUSTERED ([MD5URL] ASC)
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


GO
/****** Object:  Table [dbo].[URLScheme]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[URLScheme] (
	[SID] [smallint] IDENTITY(1, 1) NOT NULL,
	[Scheme] [nvarchar] (10) NULL,
	[CDes] [nvarchar] (255) NULL,
	CONSTRAINT [PK_URLScheme] PRIMARY KEY CLUSTERED ([SID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [UQ_URLScheme] UNIQUE NONCLUSTERED ([Scheme] ASC)
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


GO
/****** Object:  Table [dbo].[UserAgent]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[UserAgent] (
	[UAID] [int] IDENTITY(1, 1) NOT NULL,
	[UAString] [nvarchar] (900) NOT NULL,
	[Since] [datetime] NOT NULL,
	CONSTRAINT [PK_UserAgent] PRIMARY KEY CLUSTERED ([UAID] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		UNIQUE NONCLUSTERED ([UAString] ASC)
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


GO
/****** Object:  Table [dbo].[VerificationToken]    Script Date: 2024/12/3 下午 03:27:13 ******/
SET
ANSI_NULLS ON;


GO
SET
QUOTED_IDENTIFIER ON;


GO
CREATE TABLE [dbo].[VerificationToken] (
	[identifier] [nvarchar] (128) NOT NULL,
	[token] [nvarchar] (128) NOT NULL,
	[expires] [datetime2] (7) NOT NULL,
	CONSTRAINT [VerificationToken_identifier_token_key] UNIQUE NONCLUSTERED ([identifier] ASC, [token] ASC)
	WITH
		(
			PAD_INDEX = OFF,
			STATISTICS_NORECOMPUTE = OFF,
			IGNORE_DUP_KEY = OFF,
			ALLOW_ROW_LOCKS = ON,
			ALLOW_PAGE_LOCKS = ON,
			OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF
		) ON [PRIMARY],
		CONSTRAINT [VerificationToken_token_key] UNIQUE NONCLUSTERED ([token] ASC)
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


GO
ALTER TABLE [dbo].[Archive]
ADD DEFAULT ('') FOR [Keywords];


GO
ALTER TABLE [dbo].[Archive]
ADD DEFAULT ('') FOR [IndexInfo];


GO
ALTER TABLE [dbo].[Archive]
ADD DEFAULT ((0)) FOR [ContentType];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT (getdate()) FOR [Since];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT (getdate()) FOR [LastModifiedDT];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ((0)) FOR [nObject];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ((0)) FOR [cRank];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ((0)) FOR [oRank];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ((0)) FOR [nClick];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ('') FOR [Keywords];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ((0)) FOR [bHided];


GO
ALTER TABLE [dbo].[Class]
ADD DEFAULT ((0)) FOR [bDel];


GO
ALTER TABLE [dbo].[Entity]
ADD DEFAULT ((1)) FOR [bORel];


GO
ALTER TABLE [dbo].[Entity]
ADD DEFAULT ((0)) FOR [bHided];


GO
ALTER TABLE [dbo].[Entity]
ADD DEFAULT ((0)) FOR [bDel];


GO
ALTER TABLE [dbo].[GM]
ADD DEFAULT ((2)) FOR [Role];


GO
ALTER TABLE [dbo].[Groups]
ADD DEFAULT (getdate()) FOR [Since];


GO
ALTER TABLE [dbo].[Groups]
ADD DEFAULT ((0)) FOR [bHided];


GO
ALTER TABLE [dbo].[Groups]
ADD DEFAULT ((0)) FOR [bDel];


GO
ALTER TABLE [dbo].[LogDir]
ADD DEFAULT (getdate()) FOR [VisitDate];


GO
ALTER TABLE [dbo].[LogError]
ADD DEFAULT (getdate()) FOR [ThrowDate];


GO
ALTER TABLE [dbo].[LogMan]
ADD DEFAULT (getdate()) FOR [VisitDate];


GO
ALTER TABLE [dbo].[LogManTx]
ADD DEFAULT (getdate()) FOR [VisitDate];


GO
ALTER TABLE [dbo].[LogObject]
ADD DEFAULT (getdate()) FOR [VisitDate];


GO
ALTER TABLE [dbo].[LogSearch]
ADD DEFAULT (getdate()) FOR [SearchDate];


GO
ALTER TABLE [dbo].[Member]
ADD DEFAULT ((0)) FOR [LoginCount];


GO
ALTER TABLE [dbo].[Member]
ADD DEFAULT ((0)) FOR [LoginErrCount];


GO
ALTER TABLE [dbo].[Model]
ADD CONSTRAINT [Model_Active_df] DEFAULT ((0)) FOR [Active];


GO
ALTER TABLE [dbo].[Model]
ADD CONSTRAINT [Model_Version_df] DEFAULT ((1)) FOR [Version];


GO
ALTER TABLE [dbo].[MSession]
ADD DEFAULT ((0)) FOR [MID];


GO
ALTER TABLE [dbo].[MSession]
ADD DEFAULT (getdate()) FOR [Since];


GO
ALTER TABLE [dbo].[MSession]
ADD DEFAULT (getdate()) FOR [LastModifiedDT];


GO
ALTER TABLE [dbo].[Object]
ADD DEFAULT (getdate()) FOR [Since];


GO
ALTER TABLE [dbo].[Object]
ADD DEFAULT (getdate()) FOR [LastModifiedDT];


GO
ALTER TABLE [dbo].[Object]
ADD DEFAULT ((0)) FOR [nClick];


GO
ALTER TABLE [dbo].[Object]
ADD DEFAULT ((0)) FOR [bHided];


GO
ALTER TABLE [dbo].[Object]
ADD DEFAULT ((0)) FOR [bDel];


GO
ALTER TABLE [dbo].[Permission]
ADD DEFAULT ((1)) FOR [PermissionBits];


GO
ALTER TABLE [dbo].[Post]
ADD DEFAULT (getdate()) FOR [StartDT];


GO
ALTER TABLE [dbo].[SystemConfig]
ADD DEFAULT (getdate()) FOR [Since];


GO
ALTER TABLE [dbo].[SystemConfig]
ADD DEFAULT (getdate()) FOR [LastModifiedDT];


GO
ALTER TABLE [dbo].[SystemConfig]
ADD DEFAULT ((0)) FOR [bDel];


GO
ALTER TABLE [dbo].[URL]
ADD DEFAULT ('/') FOR [Path];


GO
ALTER TABLE [dbo].[UserAgent]
ADD DEFAULT (getdate()) FOR [Since];


GO
ALTER TABLE [dbo].[App_Prediction]
WITH
	CHECK
ADD CONSTRAINT [FK_App_Prediction_OID] FOREIGN KEY ([APID]) REFERENCES [dbo].[Object] ([OID]) ON UPDATE CASCADE;


GO
ALTER TABLE [dbo].[App_Prediction] CHECK CONSTRAINT [FK_App_Prediction_OID];


GO
ALTER TABLE [dbo].[App_Schedule]
WITH
	CHECK
ADD CONSTRAINT [FK_App_Schedule_OID] FOREIGN KEY ([ASID]) REFERENCES [dbo].[Object] ([OID]) ON UPDATE CASCADE;


GO
ALTER TABLE [dbo].[App_Schedule] CHECK CONSTRAINT [FK_App_Schedule_OID];


GO
ALTER TABLE [dbo].[Archive]
WITH
	CHECK
ADD CONSTRAINT [FK_Archive_AID] FOREIGN KEY ([AID]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[Archive] CHECK CONSTRAINT [FK_Archive_AID];


GO
ALTER TABLE [dbo].[Archive]
WITH
	CHECK
ADD CONSTRAINT [FK_Archive_CTID] FOREIGN KEY ([ContentType]) REFERENCES [dbo].[ContentType] ([CTID]);


GO
ALTER TABLE [dbo].[Archive] CHECK CONSTRAINT [FK_Archive_CTID];


GO
ALTER TABLE [dbo].[Class]
WITH
	CHECK
ADD CONSTRAINT [FK_Class_OwnerMID] FOREIGN KEY ([OwnerMID]) REFERENCES [dbo].[Member] ([MID]);


GO
ALTER TABLE [dbo].[Class] CHECK CONSTRAINT [FK_Class_OwnerMID];


GO
ALTER TABLE [dbo].[Class]
WITH
	CHECK
ADD CONSTRAINT [FK_Class_Type] FOREIGN KEY ([Type]) REFERENCES [dbo].[Entity] ([EID]);


GO
ALTER TABLE [dbo].[Class] CHECK CONSTRAINT [FK_Class_Type];


GO
ALTER TABLE [dbo].[CO]
WITH
	CHECK
ADD CONSTRAINT [FK_CO_CID] FOREIGN KEY ([CID]) REFERENCES [dbo].[Class] ([CID]);


GO
ALTER TABLE [dbo].[CO] CHECK CONSTRAINT [FK_CO_CID];


GO
ALTER TABLE [dbo].[CO]
WITH
	CHECK
ADD CONSTRAINT [FK_CO_OID] FOREIGN KEY ([OID]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[CO] CHECK CONSTRAINT [FK_CO_OID];


GO
ALTER TABLE [dbo].[Data_Source]
WITH
	CHECK
ADD CONSTRAINT [FK_Data_Source_OID] FOREIGN KEY ([DSID]) REFERENCES [dbo].[Object] ([OID]) ON UPDATE CASCADE;


GO
ALTER TABLE [dbo].[Data_Source] CHECK CONSTRAINT [FK_Data_Source_OID];


GO
ALTER TABLE [dbo].[EntityM2DC]
WITH
	CHECK
ADD FOREIGN KEY ([EID]) REFERENCES [dbo].[Entity] ([EID]);


GO
ALTER TABLE [dbo].[GM]
WITH
	CHECK
ADD CONSTRAINT [FK_GM_GID] FOREIGN KEY ([GID]) REFERENCES [dbo].[Groups] ([GID]);


GO
ALTER TABLE [dbo].[GM] CHECK CONSTRAINT [FK_GM_GID];


GO
ALTER TABLE [dbo].[GM]
WITH
	CHECK
ADD CONSTRAINT [FK_GM_MID] FOREIGN KEY ([MID]) REFERENCES [dbo].[Member] ([MID]);


GO
ALTER TABLE [dbo].[GM] CHECK CONSTRAINT [FK_GM_MID];


GO
ALTER TABLE [dbo].[Inheritance]
WITH
	CHECK
ADD CONSTRAINT [FK_Inheritance_CCID] FOREIGN KEY ([CCID]) REFERENCES [dbo].[Class] ([CID]);


GO
ALTER TABLE [dbo].[Inheritance] CHECK CONSTRAINT [FK_Inheritance_CCID];


GO
ALTER TABLE [dbo].[Inheritance]
WITH
	CHECK
ADD CONSTRAINT [FK_Inheritance_PCID] FOREIGN KEY ([PCID]) REFERENCES [dbo].[Class] ([CID]);


GO
ALTER TABLE [dbo].[Inheritance] CHECK CONSTRAINT [FK_Inheritance_PCID];


GO
ALTER TABLE [dbo].[Member]
WITH
	CHECK
ADD CONSTRAINT [FK_Member_ClassID] FOREIGN KEY ([ClassID]) REFERENCES [dbo].[Class] ([CID]);


GO
ALTER TABLE [dbo].[Member] CHECK CONSTRAINT [FK_Member_ClassID];


GO
ALTER TABLE [dbo].[Member]
WITH
	CHECK
ADD CONSTRAINT [FK_Member_MID] FOREIGN KEY ([MID]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[Member] CHECK CONSTRAINT [FK_Member_MID];


GO
ALTER TABLE [dbo].[Member]
WITH
	CHECK
ADD CONSTRAINT [FK_Member_Nation] FOREIGN KEY ([Nation]) REFERENCES [dbo].[Nation] ([NID]);


GO
ALTER TABLE [dbo].[Member] CHECK CONSTRAINT [FK_Member_Nation];


GO
ALTER TABLE [dbo].[ML_Engine]
WITH
	CHECK
ADD CONSTRAINT [FK_ML_Engine_OID] FOREIGN KEY ([MLEID]) REFERENCES [dbo].[Object] ([OID]) ON UPDATE CASCADE;


GO
ALTER TABLE [dbo].[ML_Engine] CHECK CONSTRAINT [FK_ML_Engine_OID];


GO
ALTER TABLE [dbo].[Model]
WITH
	CHECK
ADD CONSTRAINT [FK_Model_OID] FOREIGN KEY ([MID]) REFERENCES [dbo].[Object] ([OID]) ON UPDATE CASCADE;


GO
ALTER TABLE [dbo].[Model] CHECK CONSTRAINT [FK_Model_OID];


GO
ALTER TABLE [dbo].[MSession]
WITH
	CHECK
ADD CONSTRAINT [FK_MSession_Member] FOREIGN KEY ([MID]) REFERENCES [dbo].[Member] ([MID]) ON UPDATE CASCADE ON DELETE CASCADE;


GO
ALTER TABLE [dbo].[MSession] CHECK CONSTRAINT [FK_MSession_Member];


GO
ALTER TABLE [dbo].[MSession]
WITH
	CHECK
ADD CONSTRAINT [FK_MSession_UA] FOREIGN KEY ([UserAgent]) REFERENCES [dbo].[UserAgent] ([UAID]);


GO
ALTER TABLE [dbo].[MSession] CHECK CONSTRAINT [FK_MSession_UA];


GO
ALTER TABLE [dbo].[OAuthAccount]
WITH
	CHECK
ADD CONSTRAINT [FK_Account_MID] FOREIGN KEY ([MID]) REFERENCES [dbo].[Member] ([MID]) ON UPDATE CASCADE ON DELETE CASCADE;


GO
ALTER TABLE [dbo].[OAuthAccount] CHECK CONSTRAINT [FK_Account_MID];


GO
ALTER TABLE [dbo].[Object]
WITH
	CHECK
ADD CONSTRAINT [FK_Object_OwnerMID] FOREIGN KEY ([OwnerMID]) REFERENCES [dbo].[Member] ([MID]);


GO
ALTER TABLE [dbo].[Object] CHECK CONSTRAINT [FK_Object_OwnerMID];


GO
ALTER TABLE [dbo].[ORel]
WITH
	CHECK
ADD CONSTRAINT [FK_ORel_OID1] FOREIGN KEY ([OID1]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[ORel] CHECK CONSTRAINT [FK_ORel_OID1];


GO
ALTER TABLE [dbo].[ORel]
WITH
	CHECK
ADD CONSTRAINT [FK_ORel_OID2] FOREIGN KEY ([OID2]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[ORel] CHECK CONSTRAINT [FK_ORel_OID2];


GO
ALTER TABLE [dbo].[Permission]
WITH
	CHECK
ADD CONSTRAINT [FK_Permission_CID] FOREIGN KEY ([CID]) REFERENCES [dbo].[Class] ([CID]);


GO
ALTER TABLE [dbo].[Permission] CHECK CONSTRAINT [FK_Permission_CID];


GO
ALTER TABLE [dbo].[Post]
WITH
	CHECK
ADD CONSTRAINT [FK_Post_PID] FOREIGN KEY ([PID]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[Post] CHECK CONSTRAINT [FK_Post_PID];


GO
ALTER TABLE [dbo].[URL]
WITH
	CHECK
ADD CONSTRAINT [FK_URL_Scheme] FOREIGN KEY ([Scheme]) REFERENCES [dbo].[URLScheme] ([SID]);


GO
ALTER TABLE [dbo].[URL] CHECK CONSTRAINT [FK_URL_Scheme];


GO
ALTER TABLE [dbo].[URL]
WITH
	CHECK
ADD CONSTRAINT [FK_URL_SID] FOREIGN KEY ([SID]) REFERENCES [dbo].[StatusCode] ([Status]);


GO
ALTER TABLE [dbo].[URL] CHECK CONSTRAINT [FK_URL_SID];


GO
ALTER TABLE [dbo].[URL]
WITH
	CHECK
ADD CONSTRAINT [FK_URL_UID] FOREIGN KEY ([UID]) REFERENCES [dbo].[Object] ([OID]);


GO
ALTER TABLE [dbo].[URL] CHECK CONSTRAINT [FK_URL_UID];


GO
ALTER TABLE [dbo].[GM]
WITH
	CHECK
ADD CHECK (
	(
		[Role] >= (0)
		AND [Role] <= (2)
	)
);


GO
