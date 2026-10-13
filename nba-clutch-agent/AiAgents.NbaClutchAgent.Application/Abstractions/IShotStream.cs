namespace AiAgents.NbaClutchAgent.Application.Abstractions;

public interface IShotStream : IDisposable
{
    Task<ShotRow?> NextClutchAsync(CancellationToken ct = default);
}