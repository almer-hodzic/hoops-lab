using AiAgents.NbaClutchAgent.Domain;
using AiAgents.NbaClutchAgent.Application;

namespace AiAgents.NbaClutchAgent.Application.Learn;

public sealed class RewardService
{
    private readonly double _baselinePpp;
    private readonly double _missPenalty;

    public RewardService(double baselinePpp, double missPenalty = 0.0)
    {
        _baselinePpp = baselinePpp;
        _missPenalty = missPenalty;
    }

    public double Calculate(DecisionAction action, ShotRow row)
    {
        if (action == DecisionAction.Shoot)
        {
            var val = row.SHOT_TYPE.StartsWith("3PT") ? 3.0 : 2.0;
            return row.SHOT_MADE ? val : _missPenalty; // 0.0 je najrealnije
        }

        // DontShoot = baseline "safe" expected value
        return _baselinePpp;
    }
}
