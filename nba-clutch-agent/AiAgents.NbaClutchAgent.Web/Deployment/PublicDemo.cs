using System.Text.Json;
using System.Threading.RateLimiting;
using Microsoft.AspNetCore.RateLimiting;

namespace AiAgents.NbaClutchAgent.Web.Deployment;

public static class PublicDemo
{
    public static void Configure(WebApplicationBuilder builder)
    {
        if (Environment.GetEnvironmentVariable("PORT") is { } port)
        {
            if (!int.TryParse(port, out var number) || number < 1 || number > 65535 || number is 8011 or 8012)
                throw new InvalidOperationException("Invalid public PORT");
            builder.WebHost.UseUrls($"http://0.0.0.0:{number}");
        }
        builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = 65536);
        builder.Services.AddRateLimiter(options =>
        {
            options.RejectionStatusCode = 429;
            // Shared budgets intentionally do not trust spoofable forwarded IP headers.
            options.GlobalLimiter = PartitionedRateLimiter.CreateChained(
                PartitionedRateLimiter.Create<HttpContext, string>(ctx =>
                    ctx.Request.Path.StartsWithSegments("/api")
                        ? RateLimitPartition.GetConcurrencyLimiter("api", _ => new ConcurrencyLimiterOptions
                            { PermitLimit = 4, QueueLimit = 0 })
                        : RateLimitPartition.GetNoLimiter("other")),
                PartitionedRateLimiter.Create<HttpContext, string>(ctx =>
                    ctx.Request.Path.StartsWithSegments("/api")
                        ? RateLimitPartition.GetFixedWindowLimiter("api", _ => new FixedWindowRateLimiterOptions
                            { PermitLimit = 240, Window = TimeSpan.FromMinutes(1), QueueLimit = 0 })
                        : RateLimitPartition.GetNoLimiter("other")),
                PartitionedRateLimiter.Create<HttpContext, string>(ctx =>
                    ctx.Request.Method == "POST" && (ctx.Request.Path.StartsWithSegments("/api/decision") ||
                        ctx.Request.Path.StartsWithSegments("/api/simulation"))
                        ? RateLimitPartition.GetFixedWindowLimiter("expensive", _ => new FixedWindowRateLimiterOptions
                            { PermitLimit = 20, Window = TimeSpan.FromMinutes(1), QueueLimit = 0 })
                        : RateLimitPartition.GetNoLimiter("other")));
        });
        builder.Services.AddHttpClient("Readiness", client => client.Timeout = TimeSpan.FromSeconds(2));
    }

    public static void Use(WebApplication app)
    {
        app.Use(async (context, next) =>
        {
            // Block every verb and subpath, before routing or any legacy service resolution.
            if (context.Request.Path.StartsWithSegments("/api/run") ||
                context.Request.Path.StartsWithSegments("/api/agent") ||
                context.Request.Path.StartsWithSegments("/api/baseline"))
            {
                context.Response.StatusCode = 404;
                return;
            }
            try { await next(context); }
            catch (BadHttpRequestException ex) when (!context.Response.HasStarted)
            {
                context.Response.StatusCode = ex.StatusCode;
                await context.Response.WriteAsJsonAsync(new { error = "Invalid or oversized request" });
            }
            catch (Exception ex) when (!context.Response.HasStarted && !context.RequestAborted.IsCancellationRequested)
            {
                app.Logger.LogError(ex, "Public API request failed");
                context.Response.StatusCode = ex is HttpRequestException ? 503 : ex is OperationCanceledException ? 504 : 500;
                await context.Response.WriteAsJsonAsync(new { error = "Service temporarily unavailable" });
            }
        });
        app.UseRateLimiter();
    }

    public static IResult Upstream(HttpResponseMessage response, string body) =>
        (int)response.StatusCode >= 500
            ? Results.Json(new { error = "Service temporarily unavailable" }, statusCode: (int)response.StatusCode)
            : Results.Content(body, "application/json", statusCode: (int)response.StatusCode);

    public static void MapReadiness(WebApplication app)
    {
        var root = Environment.GetEnvironmentVariable("APP_ROOT") ?? Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, ".."));
        app.MapGet("/ready", async (IHttpClientFactory clients, CancellationToken ct) =>
        {
            try
            {
                using var manifest = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(root, "deployment", "runtime-artifacts.json"), ct));
                foreach (var item in manifest.RootElement.GetProperty("files").EnumerateArray())
                    if (!File.Exists(Path.Combine(root, item.GetProperty("path").GetString()!)))
                        return Results.Json(new { ready = false }, statusCode: 503);
                var client = clients.CreateClient("Readiness");
                foreach (var port in new[] { 8011, 8012 })
                {
                    using var response = await client.GetAsync($"http://127.0.0.1:{port}/ping", ct);
                    if (!response.IsSuccessStatusCode) return Results.Json(new { ready = false }, statusCode: 503);
                    using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
                    if (!body.RootElement.GetProperty("ok").GetBoolean()) return Results.Json(new { ready = false }, statusCode: 503);
                }
                return Results.Ok(new { ready = true });
            }
            catch (Exception) when (!ct.IsCancellationRequested)
            {
                return Results.Json(new { ready = false }, statusCode: 503);
            }
        });
    }
}
