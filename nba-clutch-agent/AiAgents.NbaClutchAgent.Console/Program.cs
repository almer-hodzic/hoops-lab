// File: AiAgents.NbaClutchAgent.Console/Program.cs

using AiAgents.NbaClutchAgent.Application;
using AiAgents.NbaClutchAgent.Application.Learn;
using AiAgents.NbaClutchAgent.Application.ProductionAgents;
using AiAgents.NbaClutchAgent.Application.Sense;
using AiAgents.NbaClutchAgent.Application.Think;
using AiAgents.NbaClutchAgent.Domain;
using AiAgents.NbaClutchAgent.Infrastructure.Dataset;

var tsvPath = Path.Combine(AppContext.BaseDirectory, "Dataset", "shots.tsv");

Console.WriteLine("NBA Clutch Agent - Console Host");
Console.WriteLine("--------------------------------");
Console.WriteLine($"Dataset path: {tsvPath}");

// -------------------- Async Helpers --------------------

static double RewardIfShoot(ShotRow row)
{
    var val = row.SHOT_TYPE.StartsWith("3PT", StringComparison.OrdinalIgnoreCase) ? 3.0 : 2.0;
    return row.SHOT_MADE ? val : 0.0;
}

static async Task<double> ComputeBaselineClutchPppAsync(string path, int maxRows, CancellationToken ct)
{
    using var s = new CsvShotStream(path);

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

static async Task<(double avg, int n)> RunAlwaysShootAsync(string path, int maxSteps, CancellationToken ct)
{
    using var s = new CsvShotStream(path);

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

static async Task<(double avg, int n)> RunAlwaysDontShootAsync(string path, double baselinePpp, int maxSteps, CancellationToken ct)
{
    using var s = new CsvShotStream(path);

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

static async Task<(double avg, int n)> RunRandomAsync(string path, double baselinePpp, int maxSteps, CancellationToken ct)
{
    var rng = new Random(123);
    using var s = new CsvShotStream(path);

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

// -------------------- Run --------------------

using var cts = new CancellationTokenSource();
var ct = cts.Token;

// Baselines
var baseline = await ComputeBaselineClutchPppAsync(tsvPath, maxRows: 20000, ct);
Console.WriteLine($"Baseline clutch PPP (per shot) ≈ {baseline:0.000}");

var (alwaysShootAvg, alwaysShootN) = await RunAlwaysShootAsync(tsvPath, maxSteps: 20000, ct);
var (alwaysDontAvg, alwaysDontN) = await RunAlwaysDontShootAsync(tsvPath, baseline, maxSteps: 20000, ct);
var (randomAvg, randomN) = await RunRandomAsync(tsvPath, baseline, maxSteps: 20000, ct);

Console.WriteLine($"AlwaysShoot avgReward={alwaysShootAvg:0.000} (n={alwaysShootN})");
Console.WriteLine($"AlwaysDontShoot avgReward={alwaysDontAvg:0.000} (n={alwaysDontN})");
Console.WriteLine($"Random 50/50 avgReward={randomAvg:0.000} (n={randomN})");
Console.WriteLine("--------------------------------");

// Agent (TRAIN -> EVAL)
const int MaxSteps = 20000;
const int TrainSteps = 5000;

Console.WriteLine($"TRAIN until step {TrainSteps}, then FREEZE and EVAL.");
Console.WriteLine($"EVAL window: steps {TrainSteps + 1}..(end)");
Console.WriteLine("--------------------------------");

using var stream = new CsvShotStream(tsvPath);

var agent = new ClutchDecisionAgent(
    stream: stream,
    sense: new FeatureBinningService(),
    policy: new BanditPolicy { Epsilon = 0.10, MinVisitsForConfidence = 15, ReviewDelta = 0.02 },
    q: new QTable(),
    reward: new RewardService(baseline),
    trainSteps: TrainSteps
);

double totalReward = 0.0;
double trainReward = 0.0;
double evalReward = 0.0;

int steps = 0;
int shoot = 0, dont = 0, review = 0;

int trainCount = 0;
int evalCount = 0;

while (steps < MaxSteps)
{
    ct.ThrowIfCancellationRequested();

    var decision = await agent.TickAsync(ct);
    if (decision is null) break;

    steps++;
    totalReward += decision.Reward;

    if (steps <= TrainSteps)
    {
        trainReward += decision.Reward;
        trainCount++;
    }
    else
    {
        evalReward += decision.Reward;
        evalCount++;
    }

    switch (decision.Action)
    {
        case DecisionAction.Shoot: shoot++; break;
        case DecisionAction.DontShoot: dont++; break;
        default: review++; break;
    }

    if (steps % 500 == 0)
    {
        var phase = steps <= TrainSteps ? "TRAIN" : "EVAL";
        Console.WriteLine(
            $"[{phase}] Step={steps} avgReward={totalReward / Math.Max(1, steps):0.000} | " +
            $"shoot={shoot} dont={dont} review={review} | " +
            $"last={decision.Action} r={decision.Reward:0.0} [{decision.Explanation}]"
        );
    }
}

Console.WriteLine("--------------------------------");

var trainAvg = trainCount == 0 ? 0.0 : trainReward / trainCount;
var evalAvg = evalCount == 0 ? 0.0 : evalReward / evalCount;
var totalAvg = steps == 0 ? 0.0 : totalReward / steps;

Console.WriteLine($"RESULTS: TrainAvg={trainAvg:0.000} (n={trainCount}) | EvalAvg={evalAvg:0.000} (n={evalCount})");
Console.WriteLine($"COMPARE: AgentTotal={totalAvg:0.000} | AlwaysShoot={alwaysShootAvg:0.000} | AlwaysDontShoot={alwaysDontAvg:0.000} | Random={randomAvg:0.000} | BaselinePPP={baseline:0.000}");

Console.WriteLine(
    $"DONE. steps={steps} avgReward={totalAvg:0.000} " +
    $"shoot%={(shoot * 100.0 / Math.Max(1, steps)):0.0}% " +
    $"review%={(review * 100.0 / Math.Max(1, steps)):0.0}%"
);
