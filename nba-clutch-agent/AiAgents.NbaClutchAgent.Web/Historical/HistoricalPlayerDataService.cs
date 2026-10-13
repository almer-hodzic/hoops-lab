using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.VisualBasic.FileIO;

namespace AiAgents.NbaClutchAgent.Web.Historical;

public interface IHistoricalPlayerDataService
{
    IReadOnlyList<HistoricalPlayerSearchResult> SearchPlayers(string? search, int limit);
    HistoricalPlayerProfile? GetPlayer(string playerId);
    HistoricalPlayerSeasonProfile? GetPlayerSeason(string playerId, string season);
    HistoricalComparison? Compare(string playerA, string seasonA, string playerB, string seasonB);
}

public sealed class HistoricalPlayerDataService : IHistoricalPlayerDataService
{
    private readonly Lazy<HistoricalIndexes> _indexes;

    public HistoricalPlayerDataService(IWebHostEnvironment environment)
    {
        _indexes = new Lazy<HistoricalIndexes>(() => LoadIndexes(environment.ContentRootPath), true);
    }

    public IReadOnlyList<HistoricalPlayerSearchResult> SearchPlayers(string? search, int limit)
    {
        var indexes = _indexes.Value;
        var normalized = Normalize(search ?? "");
        var take = Math.Clamp(limit, 1, 25);
        IEnumerable<HistoricalPlayerAggregate> query = indexes.PlayersById.Values;

        if (!string.IsNullOrWhiteSpace(normalized))
        {
            query = query
                .Where(player => player.NormalizedName.Contains(normalized, StringComparison.OrdinalIgnoreCase))
                .OrderBy(player => SearchRank(player, normalized))
                .ThenByDescending(player => player.Seasons.Count)
                .ThenBy(player => player.PlayerName);
        }
        else
        {
            query = query.OrderBy(player => player.PlayerName);
        }

        return query.Take(take).Select(ToSearchResult).ToList();
    }

    public HistoricalPlayerProfile? GetPlayer(string playerId)
    {
        if (!_indexes.Value.PlayersById.TryGetValue(playerId, out var player))
        {
            return null;
        }

        return new HistoricalPlayerProfile(
            player.PlayerId,
            player.PlayerName,
            player.PrimaryPosition,
            player.FirstSeason,
            player.LastSeason,
            player.Seasons.Count,
            player.Seasons
                .OrderBy(row => row.SeasonEndYear)
                .Select(row => new HistoricalSeasonSummary(
                    row.Season,
                    row.SeasonEndYear,
                    row.Team,
                    row.TeamName,
                    row.Position,
                    row.Age))
                .ToList());
    }

    public HistoricalPlayerSeasonProfile? GetPlayerSeason(string playerId, string season)
    {
        return _indexes.Value.SeasonsByKey.TryGetValue((playerId, season), out var row) ? ToProfile(row) : null;
    }

    public HistoricalComparison? Compare(string playerA, string seasonA, string playerB, string seasonB)
    {
        var first = GetPlayerSeason(playerA, seasonA);
        var second = GetPlayerSeason(playerB, seasonB);
        return first is null || second is null ? null : new HistoricalComparison(first, second);
    }

    private static HistoricalPlayerSearchResult ToSearchResult(HistoricalPlayerAggregate player)
    {
        return new HistoricalPlayerSearchResult(
            player.PlayerId,
            player.PlayerName,
            player.PrimaryPosition,
            player.FirstSeason,
            player.LastSeason,
            player.Seasons.Count);
    }

    private static int SearchRank(HistoricalPlayerAggregate player, string normalizedQuery)
    {
        if (player.NormalizedName.Equals(normalizedQuery, StringComparison.OrdinalIgnoreCase))
        {
            return 0;
        }

        var parts = player.NormalizedName.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length > 0 && parts[^1].Equals(normalizedQuery, StringComparison.OrdinalIgnoreCase))
        {
            return 1;
        }

        if (player.NormalizedName.StartsWith(normalizedQuery, StringComparison.OrdinalIgnoreCase))
        {
            return 2;
        }

        return 3;
    }

    private static HistoricalPlayerSeasonProfile ToProfile(HistoricalPlayerSeason row)
    {
        var games = row.Games;
        var pointsPerGame = SafeDiv(row.Points, games);
        var assistsPerGame = SafeDiv(row.Assists, games);
        var reboundsPerGame = SafeDiv(row.TotalRebounds, games);
        var stealsPerGame = SafeDiv(row.Steals, games);
        var blocksPerGame = SafeDiv(row.Blocks, games);

        return new HistoricalPlayerSeasonProfile(
            new HistoricalIdentity(
                row.PlayerId,
                row.PlayerName,
                row.Season,
                row.SeasonEndYear,
                row.Team,
                row.TeamName,
                row.FranchiseId,
                row.Age,
                row.Position,
                row.Games,
                row.GamesStarted,
                row.Minutes,
                row.MinutesPerGame,
                row.RowType,
                row.IsMultiTeamTotal),
            new HistoricalScoring(
                row.Points,
                pointsPerGame,
                row.PointsPer75,
                row.UsagePct,
                row.FieldGoalAttempts,
                row.TwoPointAttempts,
                row.ThreePointAttempts,
                row.FreeThrowAttempts,
                row.ThreePointAttemptRate,
                row.FreeThrowRate),
            new HistoricalShooting(
                SafeDiv(row.FieldGoals, row.FieldGoalAttempts),
                SafeDiv(row.TwoPointMakes, row.TwoPointAttempts),
                SafeDiv(row.ThreePointMakes, row.ThreePointAttempts),
                SafeDiv(row.FreeThrows, row.FreeThrowAttempts),
                row.TrueShootingPct,
                row.LeagueTrueShootingPct,
                row.RelativeTrueShooting,
                row.ThreePointAttemptRate,
                row.FreeThrowRate),
            new HistoricalPlaymaking(
                row.Assists,
                assistsPerGame,
                row.AssistsPer75,
                row.AssistPct,
                row.Turnovers,
                row.TurnoversPer75,
                row.TurnoverPct),
            new HistoricalRebounding(
                row.OffensiveRebounds,
                row.DefensiveRebounds,
                row.TotalRebounds,
                reboundsPerGame,
                row.ReboundsPer75,
                null,
                null,
                row.TotalReboundPct),
            new HistoricalDefense(
                row.Steals,
                stealsPerGame,
                row.StealsPer75,
                null,
                row.Blocks,
                blocksPerGame,
                row.BlocksPer75,
                null),
            new HistoricalEraContext(
                row.TeamPace,
                row.LeaguePace,
                row.LeaguePoints,
                row.LeaguePointsPer75,
                row.EstimatedPossessions,
                row.PossessionEstimateMethod,
                row.ComparisonSample,
                row.EraMetrics),
            new HistoricalAdvanced(
                row.PlayerEfficiencyRating,
                row.WinShares,
                null,
                row.BoxPlusMinus,
                null,
                null,
                null,
                "Source-derived Basketball-Reference advanced metrics where available; HoopsLab-derived fields are exposed in scoring, shooting, and era context."),
            row.Availability);
    }

    private static HistoricalIndexes LoadIndexes(string contentRoot)
    {
        var playerSeasonPath = ResolveHistoricalPath(contentRoot, "player_seasons.csv");
        var rows = ReadCsv(playerSeasonPath).Select(ToSeason).ToList();
        var players = rows
            .GroupBy(row => row.PlayerId)
            .Select(group => new HistoricalPlayerAggregate(group.Key, group.OrderBy(row => row.SeasonEndYear).ToList()))
            .ToDictionary(player => player.PlayerId, StringComparer.OrdinalIgnoreCase);
        var seasons = rows.ToDictionary(row => (row.PlayerId, row.Season), row => row);
        return new HistoricalIndexes(players, seasons);
    }

    private static string ResolveHistoricalPath(string contentRoot, string fileName)
    {
        var start = new DirectoryInfo(contentRoot);
        for (var dir = start; dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "data", "historical", "processed", fileName);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException($"Could not find data/historical/processed/{fileName} from {contentRoot}.");
    }

    private static IEnumerable<Dictionary<string, string>> ReadCsv(string path)
    {
        using var parser = new TextFieldParser(path, Encoding.UTF8)
        {
            TextFieldType = FieldType.Delimited,
            HasFieldsEnclosedInQuotes = true,
        };
        parser.SetDelimiters(",");
        var headers = parser.ReadFields() ?? [];
        while (!parser.EndOfData)
        {
            var fields = parser.ReadFields();
            if (fields is null)
            {
                continue;
            }

            var row = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < headers.Length; i++)
            {
                row[headers[i]] = i < fields.Length ? fields[i] : "";
            }
            yield return row;
        }
    }

    private static HistoricalPlayerSeason ToSeason(IReadOnlyDictionary<string, string> row)
    {
        return new HistoricalPlayerSeason(
            Get(row, "player_id") ?? "",
            Get(row, "player_name") ?? "",
            Get(row, "season") ?? "",
            (int)(GetDouble(row, "season_end_year") ?? 0),
            Get(row, "team") ?? "",
            Get(row, "team_name") ?? "",
            Get(row, "franchise_id") ?? "",
            GetDouble(row, "age"),
            NullIfBlank(Get(row, "position")),
            GetDouble(row, "games"),
            GetDouble(row, "games_started"),
            GetDouble(row, "minutes"),
            GetDouble(row, "minutes_per_game"),
            GetDouble(row, "points"),
            GetDouble(row, "field_goals"),
            GetDouble(row, "field_goal_attempts"),
            GetDouble(row, "two_point_makes"),
            GetDouble(row, "two_point_attempts"),
            GetDouble(row, "three_point_makes"),
            GetDouble(row, "three_point_attempts"),
            GetDouble(row, "free_throws"),
            GetDouble(row, "free_throw_attempts"),
            GetDouble(row, "offensive_rebounds"),
            GetDouble(row, "defensive_rebounds"),
            GetDouble(row, "total_rebounds"),
            GetDouble(row, "assists"),
            GetDouble(row, "steals"),
            GetDouble(row, "blocks"),
            GetDouble(row, "turnovers"),
            GetDouble(row, "true_shooting_pct"),
            GetDouble(row, "assist_pct"),
            GetDouble(row, "turnover_pct"),
            GetDouble(row, "total_rebound_pct"),
            GetDouble(row, "usage_pct"),
            GetDouble(row, "player_efficiency_rating"),
            GetDouble(row, "win_shares"),
            GetDouble(row, "box_plus_minus"),
            GetDouble(row, "team_pace"),
            GetDouble(row, "league_pace"),
            GetDouble(row, "league_ts_pct"),
            GetDouble(row, "league_points"),
            GetDouble(row, "league_points_per_75"),
            GetDouble(row, "estimated_possessions_minutes_team_pace"),
            NullIfBlank(Get(row, "possession_estimate_method")),
            GetDouble(row, "points_per_75"),
            GetDouble(row, "assists_per_75"),
            GetDouble(row, "rebounds_per_75"),
            GetDouble(row, "turnovers_per_75"),
            GetDouble(row, "steals_per_75"),
            GetDouble(row, "blocks_per_75"),
            GetDouble(row, "three_point_attempt_rate"),
            GetDouble(row, "free_throw_rate"),
            GetDouble(row, "relative_ts"),
            NullIfBlank(Get(row, "comparison_sample")),
            Get(row, "row_type") ?? "",
            GetBool(row, "is_multi_team_total"),
            ParseAvailability(Get(row, "availability")),
            EraMetrics(row));
    }

    private static Dictionary<string, HistoricalEraMetric> EraMetrics(IReadOnlyDictionary<string, string> row)
    {
        var metrics = new Dictionary<string, HistoricalEraMetric>(StringComparer.OrdinalIgnoreCase);
        AddEraMetric(metrics, row, "points_per_75", "Points / 75");
        AddEraMetric(metrics, row, "relative_ts", "Relative TS");
        AddEraMetric(metrics, row, "assist_pct", "AST%");
        AddEraMetric(metrics, row, "total_rebound_pct", "TRB%");
        AddEraMetric(metrics, row, "usage_pct", "USG%");
        AddEraMetric(metrics, row, "steals_per_75", "Steals / 75");
        AddEraMetric(metrics, row, "blocks_per_75", "Blocks / 75");
        AddEraMetric(metrics, row, "turnovers_per_75", "Turnovers / 75");
        return metrics;
    }

    private static void AddEraMetric(Dictionary<string, HistoricalEraMetric> metrics, IReadOnlyDictionary<string, string> row, string key, string label)
    {
        metrics[label] = new HistoricalEraMetric(
            GetDouble(row, key),
            GetDouble(row, $"{key}_season_percentile"),
            GetDouble(row, $"{key}_season_z"));
    }

    private static Dictionary<string, string?> ParseAvailability(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return new Dictionary<string, string?>();
        }

        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, string?>>(value) ?? new Dictionary<string, string?>();
        }
        catch (JsonException)
        {
            return new Dictionary<string, string?>();
        }
    }

    private static string? Get(IReadOnlyDictionary<string, string> row, string key)
    {
        return row.TryGetValue(key, out var value) ? value : null;
    }

    private static string? NullIfBlank(string? value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : value;
    }

    private static double? GetDouble(IReadOnlyDictionary<string, string> row, string key)
    {
        if (!row.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var result)
            ? result
            : null;
    }

    private static bool GetBool(IReadOnlyDictionary<string, string> row, string key)
    {
        return row.TryGetValue(key, out var value) && bool.TryParse(value, out var result) && result;
    }

    private static double? SafeDiv(double? numerator, double? denominator)
    {
        return numerator is null || denominator is null || denominator == 0 ? null : numerator / denominator;
    }

    private static string Normalize(string value)
    {
        var normalized = value.Normalize(NormalizationForm.FormD);
        var builder = new StringBuilder(normalized.Length);
        foreach (var ch in normalized)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark)
            {
                builder.Append(char.ToLowerInvariant(ch));
            }
        }
        return builder.ToString().Normalize(NormalizationForm.FormC);
    }

    private sealed record HistoricalIndexes(
        Dictionary<string, HistoricalPlayerAggregate> PlayersById,
        Dictionary<(string PlayerId, string Season), HistoricalPlayerSeason> SeasonsByKey);

    private sealed class HistoricalPlayerAggregate
    {
        public HistoricalPlayerAggregate(string playerId, IReadOnlyList<HistoricalPlayerSeason> seasons)
        {
            PlayerId = playerId;
            Seasons = seasons;
            PlayerName = seasons.Last().PlayerName;
            NormalizedName = Normalize(PlayerName);
            PrimaryPosition = seasons
                .Where(row => !string.IsNullOrWhiteSpace(row.Position))
                .GroupBy(row => row.Position)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.Key)
                .FirstOrDefault()?.Key;
            FirstSeason = seasons.First().Season;
            LastSeason = seasons.Last().Season;
        }

        public string PlayerId { get; }
        public string PlayerName { get; }
        public string NormalizedName { get; }
        public string? PrimaryPosition { get; }
        public string FirstSeason { get; }
        public string LastSeason { get; }
        public IReadOnlyList<HistoricalPlayerSeason> Seasons { get; }
    }

    private sealed record HistoricalPlayerSeason(
        string PlayerId,
        string PlayerName,
        string Season,
        int SeasonEndYear,
        string Team,
        string TeamName,
        string FranchiseId,
        double? Age,
        string? Position,
        double? Games,
        double? GamesStarted,
        double? Minutes,
        double? MinutesPerGame,
        double? Points,
        double? FieldGoals,
        double? FieldGoalAttempts,
        double? TwoPointMakes,
        double? TwoPointAttempts,
        double? ThreePointMakes,
        double? ThreePointAttempts,
        double? FreeThrows,
        double? FreeThrowAttempts,
        double? OffensiveRebounds,
        double? DefensiveRebounds,
        double? TotalRebounds,
        double? Assists,
        double? Steals,
        double? Blocks,
        double? Turnovers,
        double? TrueShootingPct,
        double? AssistPct,
        double? TurnoverPct,
        double? TotalReboundPct,
        double? UsagePct,
        double? PlayerEfficiencyRating,
        double? WinShares,
        double? BoxPlusMinus,
        double? TeamPace,
        double? LeaguePace,
        double? LeagueTrueShootingPct,
        double? LeaguePoints,
        double? LeaguePointsPer75,
        double? EstimatedPossessions,
        string? PossessionEstimateMethod,
        double? PointsPer75,
        double? AssistsPer75,
        double? ReboundsPer75,
        double? TurnoversPer75,
        double? StealsPer75,
        double? BlocksPer75,
        double? ThreePointAttemptRate,
        double? FreeThrowRate,
        double? RelativeTrueShooting,
        string? ComparisonSample,
        string RowType,
        bool IsMultiTeamTotal,
        IReadOnlyDictionary<string, string?> Availability,
        IReadOnlyDictionary<string, HistoricalEraMetric> EraMetrics);
}
