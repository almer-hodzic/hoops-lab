using AiAgents.NbaClutchAgent.Application.Abstractions;
using AiAgents.NbaClutchAgent.Application.Learn;
using AiAgents.NbaClutchAgent.Application.Sense;
using AiAgents.NbaClutchAgent.Application.Think;
using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.ProductionAgents;

public sealed class ClutchDecisionAgent : ITickAgent<ShotDecision>
{
    private readonly IShotStream _stream;
    private readonly FeatureBinningService _sense;
    private readonly BanditPolicy _policy;
    private readonly QTable _q;
    private readonly RewardService _reward;

    private int _step = 0;
    private readonly int _trainSteps;

    public bool IsFrozen => _step >= _trainSteps;

    public ShotDecision? LastDecision { get; private set; }

    public ClutchDecisionAgent(
        IShotStream stream,
        FeatureBinningService sense,
        BanditPolicy policy,
        QTable q,
        RewardService reward,
        int trainSteps = 5000)
    {
        _stream = stream;
        _sense = sense;
        _policy = policy;
        _q = q;
        _reward = reward;
        _trainSteps = trainSteps;
    }

    public async Task<ShotDecision?> TickAsync(CancellationToken ct = default)
    {
        // 1) Sense: uzmi sljedeći clutch događaj
        var row = await _stream.NextClutchAsync(ct);
        if (row is null) return null;

        // 2) Sense: raw -> Observation
        var obs = _sense.Build(row);

        // 3) Think: odaberi akciju (policy može vratiti Review)
        var action = _policy.Decide(obs, _q, out var why);

        // ✅ FIX: Review ne smije biti "izvršena" akcija, inače nema učenja.
        // Review je samo label/objašnjenje. Izvrši konkretno Shoot/Dont.
        var executedAction = action == DecisionAction.Review
            ? ResolveReviewAction(obs)
            : action;

        // 4) Learn: reward + (ako nije frozen) update Q
        var reward = _reward.Calculate(executedAction, row);

        if (!IsFrozen)
            _q.Update(obs, executedAction, reward);

        _step++;

        var explanation = action == DecisionAction.Review
            ? $"{why} | EXECUTE -> {executedAction}"
            : why;

        // ShotDecision(ShotObservation, DecisionAction planned, DecisionAction executed, ...)
        var decision = new ShotDecision(
            Observation: obs,
            Action: action,
            ExecutedAction: executedAction,
            Reward: reward,
            Explanation: explanation,
            StepIndex: _step,
            LocX: row.LOC_X,
            LocY: row.LOC_Y,
            ShotMade: row.SHOT_MADE,
            ShotType: row.SHOT_TYPE
        );

        LastDecision = decision;
        return decision;
    }

    private DecisionAction ResolveReviewAction(ShotObservation s)
    {
        // Deterministic: izvrši akciju sa većim Q; ako je izjednačeno, random 50/50.
        var e = _q.Get(s);

        if (e.QShoot > e.QDont) return DecisionAction.Shoot;
        if (e.QDont > e.QShoot) return DecisionAction.DontShoot;

        return Random.Shared.Next(2) == 0 ? DecisionAction.Shoot : DecisionAction.DontShoot;
    }
}