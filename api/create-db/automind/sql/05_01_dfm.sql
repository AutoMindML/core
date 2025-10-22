if not exists (
	select * from INFORMATION_SCHEMA.TABLES 
	where 
		TABLE_NAME = 'DFM'
		and TABLE_SCHEMA = 'dbo'
)
begin
create table dbo.DFM (
	DID int identity not null,
	Datasets nvarchar(max) default '',
	PKs nvarchar(max) default '',
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
	@message nvarchar(4000) output,
	@new_id int output
)
as begin try
	begin tran;

	-- 116 = data:fusion
	insert into dbo.[Object] (CName, CDes, [Type], OwnerMID, DataByte)
	values (@name, @des, 116, @user_id, 0);

	select @new_id = scope_identity();

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
	declare @error_message nvarchar(4000) = error_message();
	raiserror (@error_message, 18, 1);
end catch;

go

create or alter procedure dbo.xp_update_dfm (
	@user_id int,
	@fusion_id int,
	@dataset_ids nvarchar(max),
	@relationships nvarchar(max),
	@target_dataset_id int,
	@primary_keys nvarchar(max),
	@state int output,
	@message nvarchar(4000) output,
	@new_id int output
)
as begin try
	begin tran;

	if not exists (select * from [Object] where OwnerMID = @user_id and OID = @fusion_id and [Type] = 116)
	begin
		select
			@state = 1,
			@message = 'data fusion id not exists | this data is not for fusion task | user has no permission';
		commit tran;
		return;
	end

	if exists (
		select 
			D.oid
		from 
			string_split(@dataset_ids, ',', 1) dataset
			left join vd_Data_Source D on D.oid = try_cast(dataset.[value] as int) and owner_mid = @user_id
		where 
			D.oid is null
	)
	begin
		select
			@state = 1,
			@message = 'given dataset ids have invalid dataset id or user has no permission for this dataset';
		commit tran;
		return;
	end

	if not exists (
		select 
			dataset.[value]
		from 
			string_split(@dataset_ids, ',', 1) dataset
		where 
			@target_dataset_id = dataset.[value]
	)
	begin
		select
			@state = 1,
			@message = 'target dataset id must in dataset list';
		commit tran;
		return;
	end

	if not exists (
		select * from vd_Data_Source D where D.oid = @target_dataset_id and owner_mid = @user_id 
	)
	begin
		select
			@state = 1,
			@message = 'target dataset id is not exists or user has no permission';
		commit tran;
		return;
	end

	update dbo.DFM
		set Datasets = @dataset_ids
			, Relationships = @relationships
			, TargetDataset = @target_dataset_id
			, PKs = @primary_keys
		where DID = @fusion_id;

	select
		@state = 0,
		@message = 'update dfm successfully';
	commit tran;
end try
begin catch;
	if @@TRANCOUNT > 0 rollback tran;
	declare @error_message nvarchar(4000) = error_message();
	raiserror (@error_message, 18, 1);
end catch;

go

create or alter view dbo.vd_data_fusion
as
select
	DID as fusion_id
	, Datasets as dataset_ids
	, Relationships as relationships
	, TargetDataset as target_dataset_id
	, Pks as primary_keys
from dbo.DFM

go

create or alter procedure dbo.xp_data_fusion_merge (
	@fusion_id int,
	@user_id int,
	@md5 varchar(32),
	@state int output,
	@message nvarchar(4000) output,
	@new_id int output
)
as begin try
	begin tran;

	declare @binary_md5 binary(16) = convert(binary(16), @md5, 2);
	set @md5 = convert(varchar(32), @binary_md5, 2);

	if not exists (
		select
			MD5
		from
			[Data_Source]
		where
			MD5 = @binary_md5
	)
	INSERT INTO
		[Data_Source] (DSID, MD5, ConnectionData)
	VALUES
		(@fusion_id, @binary_md5, NULL);


	update [Object]
		set
			EName = @md5
		where
			OID = @fusion_id;

	
	select
		@state = 0,
		@message = 'merge data fusion successfully';

	commit tran;
	end try
begin catch;
	if @@TRANCOUNT > 0 rollback tran;
	declare @error_message nvarchar(4000) = error_message();
	raiserror (@error_message, 18, 1);
end catch;

go