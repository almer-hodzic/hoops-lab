namespace AiAgents.NbaClutchAgent.Application.Abstractions;

public interface IShotStreamFactory
{
    IShotStream Create(string datasetPath);
}