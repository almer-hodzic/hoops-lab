namespace AiAgents.NbaClutchAgent.Application.Run;

public sealed record RunRequest(int TrainSteps = 5000, int MaxSteps = 20000);

public sealed record BaselineResponse(string Dataset, double BaselinePpp);

public sealed record CountsDto(int Shoot, int Dont, int Review = 0);

public sealed record DecisionDto(
    int Step,
    string Action,
    string ExecutedAction,
    double Reward,
    string Explanation,
    string Observation
);

public sealed record MetaDto(
    string Dataset,
    int TrainSteps,
    int MaxSteps,
    double Epsilon,
    double MissPenalty,
    object BaselinesN
);

public sealed record RunResponse(
    double BaselinePpp,

    double AlwaysShootAvg,
    double AlwaysDontShootAvg,
    double RandomAvg,
    double Greedy3PtAvg,

    double AgentTotalAvg,
    double TrainAvg,
    double EvalAvg,

    int Steps,

    CountsDto Counts,

    object ActionShare,

    DecisionDto? LastDecision,

    MetaDto Meta
);