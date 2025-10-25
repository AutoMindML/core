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
	SourceUpdated datetime null,
	constraint PK_Meta_MID primary key clustered (MID ASC),
	constraint FK_Meta_MID foreign key (MID) references [Object] (OID)
);
end;
go

create or alter view vd_metadata
as
select
	MID as metadata_id
	, Prompt as prompt
	, SourceUpdated as source_updated
from dbo.MetaData

go

create or alter procedure dbo.xp_add_metadata (
	@dataset_id int,
	@user_id int,
	@prompt nvarchar(max),
	@state int output,
	@message nvarchar(4000) output,
	@new_id int output
)
as begin try

	if not exists (select 1 from vd_Data_Source where oid = @dataset_id and owner_mid = @user_id)
	begin
		select
			@state = 0,
			@message = 'data not exists or user has no permission';
		return
	end

	begin tran;


		merge into [dbo].[MetaData] as t
		using (
			values (@dataset_id, @prompt, (select LastModifiedDT from [Object] where OID = @dataset_id))
		) as s (metadata_id, prompt, source_updated)
		on s.metadata_id = t.MID
		when matched
			then
				update
				set t.Prompt = s.prompt
		when not matched by target
			then
				insert (MID, Prompt, SourceUpdated)
				values (s.metadata_id, s.prompt, s.source_updated);

	commit tran;

	select
		@state = 0,
		@message = 'add metadata successfully'

end try
begin catch;
	if @@TRANCOUNT > 0 rollback tran;
	declare 
		@error_message nvarchar(4000) = error_message()
		, @error_severity int = error_severity()
		, @error_state int = error_state();
	raiserror (@error_message, @error_severity, @error_state);
end catch;

go