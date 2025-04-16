using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GoalReached : MonoBehaviour
{
    [System.NonSerialized] public bool pathExists = true;
    [System.NonSerialized] public bool timedOut = false;

    [System.NonSerialized] public float simulationTimer;//timer to calculate simulation time
    [System.NonSerialized] public float proximityTimer;//timer to calculate proximity time
    [System.NonSerialized] public bool reached;
    [System.NonSerialized] public int collisionCount;
    
    void Start()
    {
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        if (other.gameObject.CompareTag("Agent") && !reached)
        {
            reached = true;
            AgentController controller = other.gameObject.GetComponent<AgentController>();
            simulationTimer = controller.simulationTimer;
            collisionCount = controller.collisionCount;
            proximityTimer = controller.proximityTimer;
            Debug.Log($"Goal Reached; simulation time: {simulationTimer}; collision count: {collisionCount}; proximity timer: {proximityTimer}");
            other.gameObject.GetComponent<Rigidbody2D>().Sleep();
        }
    }
    public void NoPathFound(){
        pathExists = false;
        reached = true;
    }

    public void TimedOut(){
        timedOut = true;
        reached = true;
    }
    
}