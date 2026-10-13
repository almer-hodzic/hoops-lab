using AiAgents.NbaClutchAgent.Application.Runner;

namespace AiAgents.NbaClutchAgent.Web.Services;

public sealed class ClutchAgentWorker : BackgroundService
{
    private readonly IClutchDecisionRunner _runner;

    public ClutchAgentWorker(IClutchDecisionRunner runner)
    {
        _runner = runner;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var noWorkDelay = TimeSpan.FromMilliseconds(350);

        const int batchSize = 50; // ✅ 50 tickova po loop-u

        while (!stoppingToken.IsCancellationRequested)
        {
            var status = _runner.GetStatus();
            if (!status.IsRunning)
            {
                await Task.Delay(noWorkDelay, stoppingToken);
                continue;
            }

            var didAny = false;

            for (var i = 0; i < batchSize; i++)
            {
                var step = await _runner.StepAsync(stoppingToken);

                if (step.Status == RunnerStepStatus.DidWork)
                {
                    didAny = true;
                    continue;
                }

                // NoWork / Completed => izlaz iz batch-a
                break;
            }

            if (!didAny)
                await Task.Delay(noWorkDelay, stoppingToken);
        }
    }
}