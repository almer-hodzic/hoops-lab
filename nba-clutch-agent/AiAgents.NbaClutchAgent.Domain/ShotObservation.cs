namespace AiAgents.NbaClutchAgent.Domain;

public enum DistanceBin
{
    LT8,       // Less than 8 ft
    FT8_16,    // 8–16 ft
    FT16_24,   // 16–24 ft
    FT24P     // 24+ ft (3PT)
}

public enum ClockBin
{
    VeryLate,  // 0–30s
    Late       // 31–120s
}

public sealed record ShotObservation(
    DistanceBin Distance,
    ClockBin Clock,
    int ShotValue // 2 or 3
);
