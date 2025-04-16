using UnityEngine;

[System.Serializable]
public class SimulationResponse
{
    public bool pathExists;
    public bool timedOut;
    public float simulationTime;
    public int collisionCount;
    public float proximityTime;
    public SimulationResponse(bool _pathExists, bool _timedOut, float _simulationTime, int _collisionCount, float _proximityTime)
    {
        pathExists = _pathExists;
        timedOut = _timedOut;
        simulationTime = _simulationTime;
        collisionCount = _collisionCount;
        proximityTime = _proximityTime;
    }
}