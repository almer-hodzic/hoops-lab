namespace AiAgents.NbaClutchAgent.Application.Learn;

public interface IQTableStore
{
    Task<QTableState?> TryLoadAsync(CancellationToken ct = default);
    Task SaveAsync(QTableState state, CancellationToken ct = default);
}