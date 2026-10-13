using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Learn;

public sealed record QTableRow(
    ShotObservation Observation,
    double QShoot,
    double QDont,
    int NShoot,
    int NDont
);

public sealed record QTableState(
    List<QTableRow> Rows,
    DateTimeOffset SavedAtUtc
);