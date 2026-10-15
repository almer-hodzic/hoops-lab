using AiAgents.NbaClutchAgent.Application.Abstractions;
using AiAgents.NbaClutchAgent.Application.Learn;
using AiAgents.NbaClutchAgent.Application.Run;
using AiAgents.NbaClutchAgent.Application.Runner;
using AiAgents.NbaClutchAgent.Infrastructure.Dataset;
using AiAgents.NbaClutchAgent.Web.Historical;
using AiAgents.NbaClutchAgent.Web.Services;
using AiAgents.NbaClutchAgent.Web.Simulation;
using System.Text;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// -------------------- Paths (host-only) --------------------
string DatasetPath() => Path.Combine(AppContext.BaseDirectory, "Dataset", "shots.tsv");

// NOTE: Persist path should NOT live in bin\Debug long-term.
// ContentRootPath makes it stable during development (project folder).
string QTablePath() => Path.Combine(builder.Environment.ContentRootPath, "State", "qtable.json");

// -------------------- DI wiring (thin host) --------------------
// Infrastructure
builder.Services.AddSingleton<IShotStreamFactory, CsvShotStreamFactory>();
builder.Services.AddSingleton<IQTableStore>(_ => new JsonQTableStore(QTablePath()));

// Application services
builder.Services.AddSingleton<IRunService, RunService>();
builder.Services.AddSingleton<IHistoricalPlayerDataService, HistoricalPlayerDataService>();

// Runner (shared state for background loop + endpoints)
builder.Services.AddSingleton<IClutchDecisionRunner>(sp =>
{
    var streamFactory = sp.GetRequiredService<IShotStreamFactory>();
    var qStore = sp.GetRequiredService<IQTableStore>();
    return new ClutchDecisionRunner(streamFactory, qStore, DatasetPath());
});

// Background worker
builder.Services.AddHostedService<ClutchAgentWorker>();
builder.Services.AddHttpClient("ProductionInference", client =>
{
    var baseUrl = builder.Configuration["ProductionInference:BaseUrl"] ?? "http://127.0.0.1:8011";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(10);
});

builder.Services.AddSimulationApi(builder.Configuration);

var app = builder.Build();
app.MapSimulationApi();

// -------------------- Static UI --------------------
app.UseDefaultFiles();
app.UseStaticFiles();

// -------------------- Health --------------------
app.MapGet("/ping", () => Results.Ok(new { ok = true, ts = DateTime.UtcNow }));

// -------------------- Baseline (thin web -> calls Application) --------------------
app.MapGet("/api/baseline", async (IRunService run, CancellationToken ct) =>
{
    var res = await run.GetBaselineAsync(DatasetPath(), ct);
    return Results.Ok(new { dataset = res.Dataset, baselinePpp = res.BaselinePpp });
});

// -------------------- IMPORTANT: /api/run must NOT execute whole loop in request --------------------
// Keep endpoint for frontend compatibility, but it ONLY STARTS background runner.
// Frontend should poll /api/agent/status for progress/results.
app.MapPost("/api/run", async (int trainSteps, int maxSteps, IClutchDecisionRunner runner, CancellationToken ct) =>
{
    await runner.StartAsync(trainSteps, maxSteps, ct);
    return Results.Accepted("/api/agent/status", runner.GetStatus());
});

// -------------------- Background runner control --------------------
app.MapPost("/api/agent/start", async (int trainSteps, int maxSteps, IClutchDecisionRunner runner, CancellationToken ct) =>
{
    await runner.StartAsync(trainSteps, maxSteps, ct);
    return Results.Ok(runner.GetStatus());
});

app.MapPost("/api/agent/stop", (IClutchDecisionRunner runner) =>
{
    runner.Stop();
    return Results.Ok(runner.GetStatus());
});

app.MapGet("/api/agent/status", (IClutchDecisionRunner runner) =>
{
    return Results.Ok(runner.GetStatus());
});

// -------------------- Historical player explorer --------------------
// Read-only canonical historical player data from data/historical/processed.
// The service parses once and keeps indexed player/season lookups in memory.
app.MapGet("/api/historical/players", (string? search, int? limit, IHistoricalPlayerDataService historical) =>
{
    return Results.Ok(new { items = historical.SearchPlayers(search, limit ?? 12) });
});

app.MapGet("/api/historical/players/{playerId}", (string playerId, IHistoricalPlayerDataService historical) =>
{
    var profile = historical.GetPlayer(playerId);
    return profile is null ? Results.NotFound(new { error = "Historical player not found." }) : Results.Ok(profile);
});

app.MapGet("/api/historical/players/{playerId}/seasons/{season}", (string playerId, string season, IHistoricalPlayerDataService historical) =>
{
    var profile = historical.GetPlayerSeason(playerId, season);
    return profile is null ? Results.NotFound(new { error = "Historical player season not found." }) : Results.Ok(profile);
});

app.MapGet("/api/historical/compare", (string playerA, string seasonA, string playerB, string seasonB, IHistoricalPlayerDataService historical) =>
{
    var comparison = historical.Compare(playerA, seasonA, playerB, seasonB);
    return comparison is null ? Results.NotFound(new { error = "One or both historical player seasons were not found." }) : Results.Ok(comparison);
});

// -------------------- Production decision intelligence --------------------
// These routes are intentionally thin: the frozen sklearn model, calibration,
// player profiles, and Decision Policy v1 config live in the local Python
// inference service so prediction parity stays exact.
static async Task<IResult> ProxyJsonAsync(HttpClient client, HttpRequest request, string path, CancellationToken ct)
{
    using var upstream = await client.GetAsync(path + request.QueryString, ct);
    var body = await upstream.Content.ReadAsStringAsync(ct);
    return Results.Content(body, "application/json", statusCode: (int)upstream.StatusCode);
}

app.MapPost("/api/decision/analyze", async (HttpRequest request, IHttpClientFactory clients, CancellationToken ct) =>
{
    var client = clients.CreateClient("ProductionInference");
    using var reader = new StreamReader(request.Body, Encoding.UTF8);
    var requestBody = await reader.ReadToEndAsync(ct);
    using var content = new StringContent(requestBody, Encoding.UTF8, "application/json");
    using var upstream = await client.PostAsync("/api/decision/analyze", content, ct);
    var body = await upstream.Content.ReadAsStringAsync(ct);
    return Results.Content(body, "application/json", statusCode: (int)upstream.StatusCode);
});

app.MapGet("/api/players", async (IHttpClientFactory clients, HttpRequest request, CancellationToken ct) =>
{
    return await ProxyJsonAsync(clients.CreateClient("ProductionInference"), request, "/api/players", ct);
});

app.MapGet("/api/players/{id}/profile", async (string id, IHttpClientFactory clients, CancellationToken ct) =>
{
    var client = clients.CreateClient("ProductionInference");
    using var upstream = await client.GetAsync($"/api/players/{Uri.EscapeDataString(id)}/profile", ct);
    var body = await upstream.Content.ReadAsStringAsync(ct);
    return Results.Content(body, "application/json", statusCode: (int)upstream.StatusCode);
});

app.MapGet("/api/analysis/decisions", async (IHttpClientFactory clients, HttpRequest request, CancellationToken ct) =>
{
    return await ProxyJsonAsync(clients.CreateClient("ProductionInference"), request, "/api/analysis/decisions", ct);
});

app.MapGet("/api/model", async (IHttpClientFactory clients, CancellationToken ct) =>
{
    var client = clients.CreateClient("ProductionInference");
    using var upstream = await client.GetAsync("/api/model", ct);
    var body = await upstream.Content.ReadAsStringAsync(ct);
    return Results.Content(body, "application/json", statusCode: (int)upstream.StatusCode);
});

app.Run();
