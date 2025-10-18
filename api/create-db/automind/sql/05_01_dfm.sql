if not exists (
	select * from INFORMATION_SCHEMA.TABLES 
	where 
		TABLE_NAME = 'DFM'
		and TABLE_SCHEMA = 'dbo'
)
begin
create table dbo.DFM (
	DID int identity not null,
	Datasets nvarchar(max) default '[]',
	Relationships nvarchar(max) default '[]',
	TargetDataset int null,
	constraint PK_DFM_DID primary key clustered (DID ASC),
	constraint FK_DFM_DID foreign key (DID) references [Object] (OID)
);
end;

go

create or alter procedure dbo.xp_init_dfm (
	@user_id int,
	@name nvarchar(512),
	@des nvarchar(4000),
	@state int output,
	@message  nvarchar(4000) output
)
as begin try
	begin tran;

	-- 116 = data:fusion
	insert into dbo.[Object] (CName, CDes, [Type], OwnerMID)
	values (@name, @des, 116, @user_id);

	declare @new_id int = scope_identity();

	set identity_insert dbo.DFM on;
	insert into dbo.DFM (DID)
	values (@new_id);
	set identity_insert dbo.DFM off;

	declare @data_source_cid int = (
		select dbo.fn_get_member_data_source_cid (@user_id)
	);

	insert into CO (CID, OID)
	values (@data_source_cid, @new_id);

	commit tran;

	select
		@state = 0,
		@message = 'init dfm successfully'

end try
begin catch;
	if @@TRANCOUNT > 0 rollback tran;
	select
		@state = 1,
		@message = ERROR_MESSAGE()
end catch;

go

create or alter procedure dbo.xp_update_dfm (
	@user_id int,
	@dfm_id int,
	@datasets nvarchar(max),
	@relationships nvarchar(max),
	@target_dataset int
)
as begin
	begin try;
	begin tran;

	if not exists (select * from [Object] where OwnerMID = @user_id and OID = @dfm_id)
		throw 50403, 'data not exists or user has no permission', 1;

	update dbo.DFM
		set Datasets = @datasets
			, Relationships = @relationships
			, TargetDataset = @target_dataset
		where DID = @dfm_id;

	commit tran;
	end try
	begin catch;
		if @@TRANCOUNT > 0 rollback tran;
		throw select error_number(), error_message(), error_state();
	end catch;
end