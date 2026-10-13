using AiAgents.NbaClutchAgent.Application.Learn;
using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Think;

public sealed class BanditPolicy
{
    private readonly Random _rng = new(42);

    public double Epsilon { get; set; } = 0.20;
    public int MinVisitsForConfidence { get; set; } = 15;
    public double ReviewDelta { get; set; } = 0.02;

    public DecisionAction Decide(ShotObservation s, QTable q, out string explanation)
    {
        var e = q.Get(s);

        // Cold start: forsiraj akcije da dobiješ posjete (učenje)
        if (e.Visits < MinVisitsForConfidence)
        {
            var a = _rng.NextDouble() < 0.5 ? DecisionAction.Shoot : DecisionAction.DontShoot;
            explanation = $"COLD_START -> {a} (visits={e.Visits}, need={MinVisitsForConfidence})";
            return a;
        }

        // Explore
        if (_rng.NextDouble() < Epsilon)
        {
            var a = _rng.NextDouble() < 0.5 ? DecisionAction.Shoot : DecisionAction.DontShoot;
            explanation = $"EXPLORE -> {a} (Qs={e.QShoot:0.000}, Qd={e.QDont:0.000}, nS={e.NShoot}, nD={e.NDont})";
            return a;
        }

        // Exploit
        var best = e.QShoot >= e.QDont ? DecisionAction.Shoot : DecisionAction.DontShoot;
        explanation = $"EXPLOIT -> {best} (Qs={e.QShoot:0.000}, Qd={e.QDont:0.000}, nS={e.NShoot}, nD={e.NDont})";
        return best;
    }
}
