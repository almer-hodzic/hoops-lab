using AiAgents.NbaClutchAgent.Domain;

namespace AiAgents.NbaClutchAgent.Application.Learn;

public sealed class QTable
{
    private readonly Dictionary<ShotObservation, Entry> _map = new();

    private readonly double _priorShoot;
    private readonly double _priorDont;

    public QTable() : this(0.0, 0.0) { }

    public QTable(double priorShoot, double priorDont)
    {
        _priorShoot = priorShoot;
        _priorDont = priorDont;
    }

    public sealed record Entry(double QShoot, double QDont, int NShoot, int NDont)
    {
        public int Visits => NShoot + NDont;
    }

    public Entry Get(ShotObservation s)
        => _map.TryGetValue(s, out var e)
            ? e
            : new Entry(_priorShoot, _priorDont, 0, 0);

    public void Update(ShotObservation s, DecisionAction action, double reward)
    {
        if (action == DecisionAction.Review) return;

        var e = Get(s);

        if (action == DecisionAction.Shoot)
        {
            var n = e.NShoot + 1;
            var q = e.QShoot + (reward - e.QShoot) / n;
            _map[s] = e with { QShoot = q, NShoot = n };
            return;
        }

        var nd = e.NDont + 1;
        var qd = e.QDont + (reward - e.QDont) / nd;
        _map[s] = e with { QDont = qd, NDont = nd };
    }

    public QTableState ExportState()
    {
        var rows = _map.Select(kv => new QTableRow(
            Observation: kv.Key,
            QShoot: kv.Value.QShoot,
            QDont: kv.Value.QDont,
            NShoot: kv.Value.NShoot,
            NDont: kv.Value.NDont
        )).ToList();

        return new QTableState(rows, DateTimeOffset.UtcNow);
    }

    public void ImportState(QTableState state)
    {
        _map.Clear();

        foreach (var r in state.Rows)
        {
            _map[r.Observation] = new Entry(r.QShoot, r.QDont, r.NShoot, r.NDont);
        }
    }
}
