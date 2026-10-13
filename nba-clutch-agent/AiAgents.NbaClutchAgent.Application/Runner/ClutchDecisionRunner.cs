// File: AiAgents.NbaClutchAgent.Application/Runner/ClutchDecisionRunner.cs

using AiAgents.NbaClutchAgent.Application.Abstractions;
using AiAgents.NbaClutchAgent.Application.Learn;
using AiAgents.NbaClutchAgent.Application.ProductionAgents;
using AiAgents.NbaClutchAgent.Application.Sense;
using AiAgents.NbaClutchAgent.Application.Think;
using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Runner;

public sealed class ClutchDecisionRunner : IClutchDecisionRunner
{
    private readonly IShotStreamFactory _streamFactory;
    private readonly IQTableStore _qStore;
    private readonly string _datasetPath;

    private readonly object _gate = new();

    private IShotStream? _stream;
    private ClutchDecisionAgent? _agent;
    private QTable? _q;

    private int _trainSteps;
    private int _maxSteps;

    private int _steps;
    private int _shoot;
    private int _dont;
    private int _review;

    private double _baselinePpp;
    private double _alwaysShootAvg;
    private double _alwaysDontShootAvg;
    private double _randomAvg;
    private double _greedy3PtAvg;
    private object _baselinesN = new { };

    private double _totalReward;
    private double _trainReward;

    private ShotDecision? _lastDecision;
    private bool _running;
    private string? _message;

    private int _sinceLastSave;
    private const int SaveEverySteps = 200;

    public ClutchDecisionRunner(IShotStreamFactory streamFactory, IQTableStore qStore, string datasetPath)
    {
        _streamFactory = streamFactory;
        _qStore = qStore;
        _datasetPath = datasetPath;
    }

    public async Task StartAsync(int trainSteps, int maxSteps, CancellationToken ct = default)
    {
        // 1) Reset state + create stream (SENSE setup)
        lock (_gate)
        {
            _trainSteps = Math.Max(0, trainSteps);
            _maxSteps = Math.Max(1, maxSteps);

            _steps = 0;
            _shoot = 0;
            _dont = 0;
            _review = 0;

            _totalReward = 0.0;
            _trainReward = 0.0;

            _baselinePpp = 0.0;
            _alwaysShootAvg = 0.0;
            _alwaysDontShootAvg = 0.0;
            _randomAvg = 0.0;
            _greedy3PtAvg = 0.0;
            _baselinesN = new { };

            _lastDecision = null;
            _message = null;
            _sinceLastSave = 0;

            _running = false;

            DisposeAgentLocked();
            _stream = _streamFactory.Create(_datasetPath);
        }

        // 2) Compute baselines once (THINK prep – world stats)
        var baseline = await ComputeBaselineClutchPppAsync(_datasetPath, maxRows: 20000, ct);
        var (alwaysShootAvg, alwaysShootN) = await RunAlwaysShootAsync(_datasetPath, maxSteps: 20000, ct);
        var (alwaysDontAvg, alwaysDontN) = await RunAlwaysDontShootAsync(_datasetPath, baseline, maxSteps: 20000, ct);
        var (randomAvg, randomN) = await RunRandomAsync(_datasetPath, baseline, maxSteps: 20000, ct);
        var (greedy3Avg, greedy3N) = await RunGreedy3PTAsync(_datasetPath, baseline, maxSteps: 20000, ct);

        // 3) Load persisted Q + create agent (LEARN state restore)
        var q = new QTable();
        var loaded = await _qStore.TryLoadAsync(ct);
        if (loaded is not null)
            q.ImportState(loaded);

        var policy = new BanditPolicy { Epsilon = 0.20 };
        var rewardSvc = new RewardService(baseline, missPenalty: 0.0);
        var sense = new FeatureBinningService();

        var agent = new ClutchDecisionAgent(_stream!, sense, policy, q, rewardSvc, trainSteps: _trainSteps);

        // 4) Publish running (ACT: start)
        lock (_gate)
        {
            _baselinePpp = baseline;
            _alwaysShootAvg = alwaysShootAvg;
            _alwaysDontShootAvg = alwaysDontAvg;
            _randomAvg = randomAvg;
            _greedy3PtAvg = greedy3Avg;
            _baselinesN = new { alwaysShootN, alwaysDontN, randomN, greedy3N };

            _q = q;
            _agent = agent;
            _running = true;

            _message = loaded is null
                ? "Running (fresh)"
                : $"Running (loaded {loaded.Rows.Count} states)";
        }
    }

    public void Stop()
    {
        lock (_gate)
        {
            _running = false;
            _message = "Stopped";
            DisposeAgentLocked();
        }
    }

    public RunnerStatus GetStatus()
    {
        lock (_gate) return SnapshotLocked();
    }

    public async Task<ClutchRunnerStepResult> StepAsync(CancellationToken ct = default)
    {
        ClutchDecisionAgent? agent;
        QTable? q;
        int maxSteps;
        bool running;

        lock (_gate)
        {
            running = _running;
            agent = _agent;
            q = _q;
            maxSteps = _maxSteps;
        }

        if (!running || agent is null || q is null)
        {
            lock (_gate)
            {
                _message = "NoWork (not running)";
                return new ClutchRunnerStepResult(RunnerStepStatus.NoWork, SnapshotLocked());
            }
        }

        if (_steps >= maxSteps)
        {
            await SaveQAsync(q, ct);
            lock (_gate)
            {
                _running = false;
                _message = "Completed (maxSteps reached)";
                DisposeAgentLocked();
                return new ClutchRunnerStepResult(RunnerStepStatus.Completed, SnapshotLocked());
            }
        }

        var decision = await agent.TickAsync(ct);
        if (decision is null)
        {
            await SaveQAsync(q, ct);
            lock (_gate)
            {
                _running = false;
                _message = "Completed (end of stream)";
                DisposeAgentLocked();
                return new ClutchRunnerStepResult(RunnerStepStatus.Completed, SnapshotLocked());
            }
        }

        lock (_gate)
        {
            _steps = decision.StepIndex;
            _lastDecision = decision;

            _totalReward += decision.Reward;
            if (_steps <= _trainSteps) _trainReward += decision.Reward;

            switch (decision.ExecutedAction)
            {
                case DecisionAction.Shoot: _shoot++; break;
                case DecisionAction.DontShoot: _dont++; break;
                default: _review++; break;
            }

            _sinceLastSave++;
            _message = "DidWork";
        }

        if (_sinceLastSave >= SaveEverySteps)
        {
            await SaveQAsync(q, ct);
            lock (_gate) { _sinceLastSave = 0; }
        }

        lock (_gate)
        {
            return new ClutchRunnerStepResult(RunnerStepStatus.DidWork, SnapshotLocked());
        }
    }

    private async Task SaveQAsync(QTable q, CancellationToken ct)
    {
        await _qStore.SaveAsync(q.ExportState(), ct);
    }

    private RunnerStatus SnapshotLocked()
    {
        var trainN = Math.Min(_trainSteps, _steps);
        var evalN = Math.Max(0, _steps - _trainSteps);
        var evalReward = _totalReward - _trainReward;

        var agentTotalAvg = _steps == 0 ? 0.0 : _totalReward / _steps;
        var trainAvg = trainN == 0 ? 0.0 : _trainReward / trainN;
        var evalAvg = evalN == 0 ? 0.0 : evalReward / evalN;

        var shootPct = _steps == 0 ? 0.0 : _shoot * 100.0 / _steps;
        var dontPct = _steps == 0 ? 0.0 : _dont * 100.0 / _steps;
        var reviewPct = _steps == 0 ? 0.0 : _review * 100.0 / _steps;

        var baselines = _baselinePpp <= 0.0
            ? null
            : new RunnerBaselinesDto(
                AlwaysShootAvg: _alwaysShootAvg,
                AlwaysDontShootAvg: _alwaysDontShootAvg,
                RandomAvg: _randomAvg,
                Greedy3PtAvg: _greedy3PtAvg,
                BaselinesN: _baselinesN
            );

        var avgs = _baselinePpp <= 0.0
            ? null
            : new RunnerAveragesDto(agentTotalAvg, trainAvg, evalAvg);

        return new RunnerStatus(
            IsRunning: _running,
            Dataset: _datasetPath,
            TrainSteps: _trainSteps,
            MaxSteps: _maxSteps,
            Steps: _steps,
            BaselinePpp: _baselinePpp <= 0.0 ? null : _baselinePpp,
            Baselines: baselines,
            Averages: avgs,
            Counts: new RunnerCounts(_shoot, _dont, _review),
            ActionShare: new ActionShareDto(shootPct, dontPct, reviewPct),
            LastDecision: _lastDecision,
            Message: _message
        );
    }

    private void DisposeAgentLocked()
    {
        try { _stream?.Dispose(); } catch { }
        _stream = null;
        _agent = null;
        _q = null;
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
