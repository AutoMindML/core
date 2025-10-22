if not exists (
	select * from INFORMATION_SCHEMA.TABLES 
	where 
		TABLE_NAME = 'MetaData'
		and TABLE_SCHEMA = 'dbo'
)
begin
create table dbo.MetaData (
	MID int not null,
	Prompt nvarchar(max) null,
	SourceLastUpdated date null,
	constraint PK_Meta_MID primary key clustered (MID ASC),
	constraint FK_Meta_MID foreign key (MID) references [Object] (OID)
);
end;
go
