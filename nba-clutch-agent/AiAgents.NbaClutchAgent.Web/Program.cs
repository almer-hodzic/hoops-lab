using AiAgents.NbaClutchAgent.Web.Historical;
using AiAgents.NbaClutchAgent.Web.Simulation;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

// The public demo has no legacy training runner or writable Q-table.
builder.Services.AddSingleton<IHistoricalPlayerDataService, HistoricalPlayerDataService>();
AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Configure(builder);
builder.Services.AddHttpClient("ProductionInference", client =>
{
    var baseUrl = builder.Configuration["ProductionInference:BaseUrl"] ?? "http://127.0.0.1:8011";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(10);
});

builder.Services.AddSimulationApi(builder.Configuration);

var app = builder.Build();
AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Use(app);
app.MapSimulationApi();

// -------------------- Static UI --------------------
app.UseDefaultFiles();
app.UseStaticFiles();

// -------------------- Health --------------------
app.MapGet("/ping", () => Results.Ok(new { ok = true, ts = DateTime.UtcNow }));
AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.MapReadiness(app);

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
    return AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Upstream(upstream, body);
}

app.MapPost("/api/decision/analyze", async (HttpRequest request, IHttpClientFactory clients, CancellationToken ct) =>
{
    var client = clients.CreateClient("ProductionInference");
    if (!request.HasJsonContentType())
        return Results.Json(new { error = "Content-Type must be application/json" }, statusCode: 415);
    using var reader = new StreamReader(request.Body, Encoding.UTF8);
    var requestBody = await reader.ReadToEndAsync(ct);
    using var content = new StringContent(requestBody, Encoding.UTF8, "application/json");
    using var upstream = await client.PostAsync("/api/decision/analyze", content, ct);
    var body = await upstream.Content.ReadAsStringAsync(ct);
    return AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Upstream(upstream, body);
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
    return AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Upstream(upstream, body);
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
    return AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Upstream(upstream, body);
});

app.Run();
