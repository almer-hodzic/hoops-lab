namespace AiAgents.NbaClutchAgent.Application.Abstractions;

public interface ITickAgent<TResult> where TResult : class
{
	Task<TResult?> TickAsync(CancellationToken ct = default);
}
