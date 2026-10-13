using AiAgents.NbaClutchAgent.Application.Abstractions;
using AiAgents.NbaClutchAgent.Application.Learn;
using AiAgents.NbaClutchAgent.Application.ProductionAgents;
using AiAgents.NbaClutchAgent.Application.Sense;
using AiAgents.NbaClutchAgent.Application.Think;
using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Run;

public sealed class RunService : IRunService
{
    private readonly IShotStreamFactory _streamFactory;
    private readonly IQTableStore _qStore;

    public RunService(IShotStreamFactory streamFactory, IQTableStore qStore)
    {
        _streamFactory = streamFactory;
        _qStore = qStore;
    }

    public async Task<BaselineResponse> GetBaselineAsync(string datasetPath, CancellationToken ct = default)
    {
        var baseline = await ComputeBaselineClutchPppAsync(datasetPath, maxRows: 20000, ct);
        return new BaselineResponse(datasetPath, baseline);
    }

    public async Task<RunResponse> RunAsync(string datasetPath, RunRequest request, CancellationToken ct = default)
    {
        var trainSteps = Math.Max(0, request.TrainSteps);
        var maxSteps = Math.Max(1, request.MaxSteps);

        var baseline = await ComputeBaselineClutchPppAsync(datasetPath, maxRows: 20000, ct);

        var (alwaysShootAvg, alwaysShootN) = await RunAlwaysShootAsync(datasetPath, maxSteps, ct);
        var (alwaysDontAvg, alwaysDontN) = await RunAlwaysDontShootAsync(datasetPath, baseline, maxSteps, ct);
        var (randomAvg, randomN) = await RunRandomAsync(datasetPath, baseline, maxSteps, ct);
        var (greedy3Avg, greedy3N) = await RunGreedy3PTAsync(datasetPath, baseline, maxSteps, ct);

        // -------------------- PERSISTED LEARNING (load previous Q) --------------------
        var q = new QTable();
        var loaded = await _qStore.TryLoadAsync(ct);
        if (loaded is not null)
            q.ImportState(loaded);

        using var stream = _streamFactory.Create(datasetPath);

        var policy = new BanditPolicy { Epsilon = 0.20 };
        var missPenalty = 0.0;
        var rewardSvc = new RewardService(baseline, missPenalty);
        var sense = new FeatureBinningService();

        var agent = new ClutchDecisionAgent(stream, sense, policy, q, rewardSvc, trainSteps);

        double totalReward = 0.0;
        double trainReward = 0.0;
        int steps = 0;

        int shoot = 0, dont = 0, review = 0;

        while (steps < maxSteps)
        {
            ct.ThrowIfCancellationRequested();

            var d = await agent.TickAsync(ct);
            if (d is null) break;

            steps++;
            totalReward += d.Reward;
            if (steps <= trainSteps) trainReward += d.Reward;

            switch (d.Action)
            {
                case DecisionAction.Shoot: shoot++; break;
                case DecisionAction.DontShoot: dont++; break;
                case DecisionAction.Review: review++; break;
            }
        }

        // -------------------- PERSISTED LEARNING (save updated Q) --------------------
        await _qStore.SaveAsync(q.ExportState(), ct);

        var trainN = Math.Min(trainSteps, steps);
        var evalN = Math.Max(0, steps - trainSteps);
        var evalReward = totalReward - trainReward;

        DecisionDto? lastDecision = agent.LastDecision is null
            ? null
            : new DecisionDto(
                Step: agent.LastDecision.StepIndex,
                Action: agent.LastDecision.Action.ToString(),
                ExecutedAction: agent.LastDecision.ExecutedAction.ToString(),
                Reward: agent.LastDecision.Reward,
                Explanation: agent.LastDecision.Explanation,
                Observation: agent.LastDecision.Observation.ToString()
            );

        var shootPct = shoot * 100.0 / Math.Max(1, steps);
        var dontPct = dont * 100.0 / Math.Max(1, steps);
        var reviewPct = review * 100.0 / Math.Max(1, steps);

        return new RunResponse(
            BaselinePpp: baseline,

            AlwaysShootAvg: alwaysShootAvg,
            AlwaysDontShootAvg: alwaysDontAvg,
            RandomAvg: randomAvg,
            Greedy3PtAvg: greedy3Avg,

            AgentTotalAvg: totalReward / Math.Max(1, steps),
            TrainAvg: trainReward / Math.Max(1, trainN),
            EvalAvg: evalReward / Math.Max(1, evalN),

            Steps: steps,

            Counts: new CountsDto(shoot, dont, review),

            ActionShare: new { shootPct, dontPct, reviewPct },

            LastDecision: lastDecision,

            Meta: new MetaDto(
                Dataset: datasetPath,
                TrainSteps: trainSteps,
                MaxSteps: maxSteps,
                Epsilon: policy.Epsilon,
                MissPenalty: missPenalty,
                BaselinesN: new { alwaysShootN, alwaysDontN, randomN, greedy3N }
            )
        );
    }

    private static double RewardIfShoot(ShotRow row)
    {
        var val = row.SHOT_TYPE.StartsWith("3PT", StringComparison.OrdinalIgnoreCase) ? 3.0 : 2.0;
        return row.SHOT_MADE ? val : 0.0;
    }

    private async Task<double> ComputeBaselineClutchPppAsync(string path, int maxRows, CancellationToken ct)
    {
        using var s = _streamFactory.Create(path);

        double sum = 0.0;
        int n = 0;

        while (n < maxRows)
        {
            ct.ThrowIfCancellationRequested();

            var row = await s.NextClutchAsync(ct);
            if (row is null) break;

            sum += RewardIfShoot(row);
            n++;
        }

        return n == 0 ? 0.95 : sum / n;
    }

    private async Task<(double avg, int n)> RunAlwaysShootAsync(string path, int maxSteps, CancellationToken ct)
    {
        using var s = _streamFactory.Create(path);

        double sum = 0.0;
        int n = 0;

        while (n < maxSteps)
        {
            ct.ThrowIfCancellationRequested();

            var row = await s.NextClutchAsync(ct);
            if (row is null) break;

            sum += RewardIfShoot(row);
            n++;
        }

        return (n == 0 ? 0.0 : sum / n, n);
    }

    private async Task<(double avg, int n)> RunAlwaysDontShootAsync(string path, double baselinePpp, int maxSteps, CancellationToken ct)
    {
        using var s = _streamFactory.Create(path);

        double sum = 0.0;
        int n = 0;

        while (n < maxSteps)
        {
            ct.ThrowIfCancellationRequested();

            var row = await s.NextClutchAsync(ct);
            if (row is null) break;

            sum += baselinePpp;
            n++;
        }

        return (n == 0 ? 0.0 : sum / n, n);
    }

    private async Task<(double avg, int n)> RunRandomAsync(string path, double baselinePpp, int maxSteps, CancellationToken ct)
    {
        var rng = new Random(123);
        using var s = _streamFactory.Create(path);

        double sum = 0.0;
        int n = 0;

        while (n < maxSteps)
        {
            ct.ThrowIfCancellationRequested();

            var row = await s.NextClutchAsync(ct);
            if (row is null) break;

            var chooseShoot = rng.NextDouble() < 0.5;
            sum += chooseShoot ? RewardIfShoot(row) : baselinePpp;
            n++;
        }

        return (n == 0 ? 0.0 : sum / n, n);
    }

    private async Task<(double avg, int n)> RunGreedy3PTAsync(string path, double baselinePpp, int maxSteps, CancellationToken ct)
    {
        using var s = _streamFactory.Create(path);

        double sum = 0.0;
        int n = 0;

        while (n < maxSteps)
        {
            ct.ThrowIfCancellationRequested();

            var row = await s.NextClutchAsync(ct);
            if (row is null) break;

            var is3 = row.SHOT_TYPE.StartsWith("3PT", StringComparison.OrdinalIgnoreCase);
            sum += is3 ? RewardIfShoot(row) : baselinePpp;
            n++;
        }

        return (n == 0 ? 0.0 : sum / n, n);
    }
}
