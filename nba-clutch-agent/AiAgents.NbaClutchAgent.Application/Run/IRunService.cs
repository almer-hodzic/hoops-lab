namespace AiAgents.NbaClutchAgent.Application.Run;

public interface IRunService
{
    Task<BaselineResponse> GetBaselineAsync(string datasetPath, CancellationToken ct = default);
    Task<RunResponse> RunAsync(string datasetPath, RunRequest request, CancellationToken ct = default);
}