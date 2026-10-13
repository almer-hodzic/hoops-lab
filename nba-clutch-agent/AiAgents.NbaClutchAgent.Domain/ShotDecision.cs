namespace AiAgents.NbaClutchAgent.Domain;

public sealed record ShotDecision(
    ShotObservation Observation,
    DecisionAction Action,
    DecisionAction ExecutedAction,
    double Reward,
    string Explanation,
    int StepIndex,
    double LocX,
    double LocY,
    bool ShotMade,
    string ShotType
);
