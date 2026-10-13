namespace AiAgents.NbaClutchAgent.Application;

// Minimalno polja koja ?itamo iz TSV/CSV
public sealed record ShotRow(
    string ZONE_RANGE,
    string SHOT_TYPE,
    bool SHOT_MADE,
    int QUARTER,
    int MINS_LEFT,
    int SECS_LEFT,
    double LOC_X,
    double LOC_Y
);
