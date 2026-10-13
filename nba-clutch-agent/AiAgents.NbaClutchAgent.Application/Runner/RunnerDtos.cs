// File: AiAgents.NbaClutchAgent.Application/Runner/RunnerDtos.cs

using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Runner;

public enum RunnerStepStatus
{
    NoWork = 0,
    DidWork = 1,
    Completed = 2
}

public sealed record RunnerCounts(int Shoot, int Dont, int Review);

public sealed record ActionShareDto(double ShootPct, double DontPct, double ReviewPct);

public sealed record RunnerBaselinesDto(
    double AlwaysShootAvg,
    double AlwaysDontShootAvg,
    double RandomAvg,
    double Greedy3PtAvg,
    object BaselinesN
);

public sealed record RunnerAveragesDto(
    double AgentTotalAvg,
    double TrainAvg,
    double EvalAvg
);

public sealed record RunnerStatus(
    bool IsRunning,
    string Dataset,
    int TrainSteps,
    int MaxSteps,
    int Steps,
    double? BaselinePpp,
    RunnerBaselinesDto? Baselines,
    RunnerAveragesDto? Averages,
    RunnerCounts Counts,
    ActionShareDto ActionShare,
    ShotDecision? LastDecision,
    string? Message
);

public sealed record ClutchRunnerStepResult(RunnerStepStatus Status, RunnerStatus StatusSnapshot);
