namespace AiAgents.NbaClutchAgent.Application.Runner;

public interface IClutchDecisionRunner
{
    Task StartAsync(int trainSteps, int maxSteps, CancellationToken ct = default);
    void Stop();
    RunnerStatus GetStatus();
    Task<ClutchRunnerStepResult> StepAsync(CancellationToken ct = default);
}