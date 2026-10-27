using System.Text;

namespace AiAgents.NbaClutchAgent.Web.Simulation;

public static class SimulationEndpoints
{
    private const int MaxBodyBytes = 65536;

    public static IServiceCollection AddSimulationApi(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddHttpClient("Simulation", client =>
        {
            client.BaseAddress = new Uri(configuration["Simulation:BaseUrl"] ?? "http://127.0.0.1:8012");
            client.Timeout = TimeSpan.FromSeconds(30);
        });
        return services;
    }

    public static void MapSimulationApi(this WebApplication app)
    {
        app.MapGet("/api/simulation/info", (IHttpClientFactory clients, CancellationToken ct) =>
            Forward(clients.CreateClient("Simulation"), new HttpRequestMessage(HttpMethod.Get, "/api/simulation/info"), ct));

        app.MapPost("/api/simulation/game", async (HttpRequest request, IHttpClientFactory clients, CancellationToken ct) =>
        {
            if (request.ContentType?.Split(';')[0].Trim() != "application/json")
                return Results.Json(new { error = "Content-Type must be application/json" }, statusCode: 415);
            if (request.ContentLength > MaxBodyBytes)
                return Results.Json(new { error = "Request body exceeds 65536 bytes" }, statusCode: 413);

            using var buffer = new MemoryStream();
            var chunk = new byte[8192];
            int read;
            while ((read = await request.Body.ReadAsync(chunk, ct)) > 0)
            {
                if (buffer.Length + read > MaxBodyBytes)
                    return Results.Json(new { error = "Request body exceeds 65536 bytes" }, statusCode: 413);
                await buffer.WriteAsync(chunk.AsMemory(0, read), ct);
            }
            var upstream = new HttpRequestMessage(HttpMethod.Post, "/api/simulation/game")
            {
                Content = new StringContent(Encoding.UTF8.GetString(buffer.ToArray()), Encoding.UTF8, "application/json")
            };
            return await Forward(clients.CreateClient("Simulation"), upstream, ct);
        });
    }

    private static async Task<IResult> Forward(HttpClient client, HttpRequestMessage message, CancellationToken ct)
    {
        using (message)
        {
            try
            {
                using var response = await client.SendAsync(message, ct);
                var body = await response.Content.ReadAsStringAsync(ct);
                return AiAgents.NbaClutchAgent.Web.Deployment.PublicDemo.Upstream(response, body);
            }
            catch (HttpRequestException)
            {
                return Results.Json(new { error = "Simulation service unavailable" }, statusCode: 503);
            }
            catch (TaskCanceledException) when (!ct.IsCancellationRequested)
            {
                return Results.Json(new { error = "Simulation service timed out" }, statusCode: 504);
            }
        }
    }
}
