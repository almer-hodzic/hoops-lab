namespace AiAgents.NbaClutchAgent.Web.Historical;

public sealed record HistoricalPlayerSearchResult(
    string PlayerId,
    string PlayerName,
    string? Position,
    string FirstSeason,
    string LastSeason,
    int NumberOfSeasons);

public sealed record HistoricalPlayerProfile(
    string PlayerId,
    string PlayerName,
    string? Position,
    string FirstSeason,
    string LastSeason,
    int NumberOfSeasons,
    IReadOnlyList<HistoricalSeasonSummary> Seasons);

public sealed record HistoricalSeasonSummary(
    string Season,
    int SeasonEndYear,
    string Team,
    string TeamName,
    string? Position,
    double? Age);

public sealed record HistoricalPlayerSeasonProfile(
    HistoricalIdentity Identity,
    HistoricalScoring Scoring,
    HistoricalShooting Shooting,
    HistoricalPlaymaking Playmaking,
    HistoricalRebounding Rebounding,
    HistoricalDefense Defense,
    HistoricalEraContext EraContext,
    HistoricalAdvanced Advanced,
    IReadOnlyDictionary<string, string?> Availability);

public sealed record HistoricalIdentity(
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
    string RowType,
    bool IsMultiTeamTotal);

public sealed record HistoricalScoring(
    double? Points,
    double? PointsPerGame,
    double? PointsPer75,
    double? UsagePct,
    double? FieldGoalAttempts,
    double? TwoPointAttempts,
    double? ThreePointAttempts,
    double? FreeThrowAttempts,
    double? ThreePointAttemptRate,
    double? FreeThrowRate);

public sealed record HistoricalShooting(
    double? FieldGoalPct,
    double? TwoPointPct,
    double? ThreePointPct,
    double? FreeThrowPct,
    double? TrueShootingPct,
    double? LeagueTrueShootingPct,
    double? RelativeTrueShooting,
    double? ThreePointAttemptRate,
    double? FreeThrowRate);

public sealed record HistoricalPlaymaking(
    double? Assists,
    double? AssistsPerGame,
    double? AssistsPer75,
    double? AssistPct,
    double? Turnovers,
    double? TurnoversPer75,
    double? TurnoverPct);

public sealed record HistoricalRebounding(
    double? OffensiveRebounds,
    double? DefensiveRebounds,
    double? TotalRebounds,
    double? ReboundsPerGame,
    double? ReboundsPer75,
    double? OffensiveReboundPct,
    double? DefensiveReboundPct,
    double? TotalReboundPct);

public sealed record HistoricalDefense(
    double? Steals,
    double? StealsPerGame,
    double? StealsPer75,
    double? StealPct,
    double? Blocks,
    double? BlocksPerGame,
    double? BlocksPer75,
    double? BlockPct);

public sealed record HistoricalEraContext(
    double? TeamPace,
    double? LeaguePace,
    double? LeaguePoints,
    double? LeaguePointsPer75,
    double? EstimatedPossessions,
    string? PossessionEstimateMethod,
    string? ComparisonSample,
    IReadOnlyDictionary<string, HistoricalEraMetric> Metrics);

public sealed record HistoricalEraMetric(
    double? Value,
    double? SeasonPercentile,
    double? SeasonZ);

public sealed record HistoricalAdvanced(
    double? PlayerEfficiencyRating,
    double? WinShares,
    double? WinSharesPer48,
    double? BoxPlusMinus,
    double? OffensiveBoxPlusMinus,
    double? DefensiveBoxPlusMinus,
    double? ValueOverReplacementPlayer,
    string Provenance);

public sealed record HistoricalComparison(
    HistoricalPlayerSeasonProfile PlayerA,
    HistoricalPlayerSeasonProfile PlayerB);
