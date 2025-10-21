create or alter procedure dbo.xp_ (

	@state int output,
	@message nvarchar(4000) output,
	@new_id int output
)
as begin try
	begin tran;

	commit tran;

	select
		@state = 0,
		@message = ''

end try
begin catch;
	if @@TRANCOUNT > 0 rollback tran;
	declare @error_message nvarchar(4000) = error_message();
	raiserror (@error_message, 18, 1);
end catch;

go