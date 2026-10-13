using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Sense;

public sealed class FeatureBinningService
{
    public ShotObservation Build(ShotRow row)
    {
        // Distance bin (koristimo ve? diskretni ZONE_RANGE)
        var distance = row.ZONE_RANGE switch
        {
            "Less Than 8 ft." => DistanceBin.LT8,
            "8-16 ft." => DistanceBin.FT8_16,
            "16-24 ft." => DistanceBin.FT16_24,
            "24+ ft." => DistanceBin.FT24P,
            _ => DistanceBin.FT16_24 // fallback
        };

        // Clock bin (clutch je ve? <=120s)
        var timeLeft = row.MINS_LEFT * 60 + row.SECS_LEFT;
        var clock = timeLeft <= 30
            ? ClockBin.VeryLate
            : ClockBin.Late;

        // Shot value
        var shotValue = row.SHOT_TYPE.StartsWith("3PT") ? 3 : 2;

        return new ShotObservation(
            Distance: distance,
            Clock: clock,
            ShotValue: shotValue
        );
    }
}
