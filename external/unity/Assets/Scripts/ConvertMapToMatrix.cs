using System;
using System.Text;
using System.Linq;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

public class ConvertMapToMatrix : MonoBehaviour
{
    public bool saveMap = true;
    public Tile[] floor;
    int vertical, horizontal, columns, rows, i = 0, j = 0, f = 0;
    int[,] matrix;
    Tile currentTile;
    Tilemap tilemap;
    StringBuilder sb;
    GameObject[] goals, agents, lowObstacles, highObstacles;
    public GameObject pathfinder;
    //public GameObject pathfinder, player;



    // Start is called before the first frame update
    void Start()
    {
        //Time.timeScale = Time.timeScale * 100;
        tilemap = GetComponent<Tilemap>();
        horizontal = (int)tilemap.localBounds.max.x;
        vertical = (int)tilemap.localBounds.max.y;

        columns = horizontal - (int)tilemap.localBounds.min.x;
        rows = vertical - (int)tilemap.localBounds.min.y;
        matrix = new int[rows, columns];

        for (int y = vertical - 1; y >= (int)tilemap.localBounds.min.y; y--)
        {
            for (int x = (int)tilemap.localBounds.min.x; x < horizontal; x++)
            {
                currentTile = (Tile)tilemap.GetTile(new Vector3Int(x, y, 0));
                if (floor.Contains(currentTile))
                {
                    matrix[i, j] = 0;
                }
                else matrix[i, j] = -1;
                j++;
            }
            i++;
            j = 0;
        }
        Debug.Log("Min x: " + tilemap.localBounds.min.x + " Min y: " + tilemap.localBounds.min.y);
        goals = GameObject.FindGameObjectsWithTag("Goal");
        agents = GameObject.FindGameObjectsWithTag("Agent");
        lowObstacles = GameObject.FindGameObjectsWithTag("Low Obstacle");
        highObstacles = GameObject.FindGameObjectsWithTag("High Obstacle");


        foreach (GameObject highObstacle in highObstacles)
        {
            matrix[(int)(-(highObstacle.transform.position.y) - tilemap.localBounds.min.y), (int)((highObstacle.transform.position.x) - tilemap.localBounds.min.x)] = 1;
        }
        foreach (GameObject lowObstacle in lowObstacles)
        {
            matrix[(int)(-(lowObstacle.transform.position.y) - tilemap.localBounds.min.y), (int)((lowObstacle.transform.position.x) - tilemap.localBounds.min.x)] = 4;
        }
        foreach (GameObject goal in goals)
        {
            matrix[(int)(-(goal.transform.position.y) - tilemap.localBounds.min.y), (int)((goal.transform.position.x) - tilemap.localBounds.min.x)] = 3;
        }
        foreach (GameObject agent in agents)
        {
            matrix[(int)(-(agent.transform.position.y) - tilemap.localBounds.min.y), (int)((agent.transform.position.x) - tilemap.localBounds.min.x)] = 2;
        }
        sb = new StringBuilder();
        for (int a = 0; a < rows; a++)
        {
            for (int b = 0; b < columns; b++)
            {
                sb.Append(matrix[a, b]);
                sb.Append(' ');
            }
            sb.AppendLine();
        }
        string pwd = System.IO.Directory.GetCurrentDirectory();
        while (System.IO.File.Exists(pwd + "/Maps/map" + f + ".txt"))
        {
            f++;
        }
        if(saveMap) {
        StreamWriter writer = new StreamWriter(pwd + "/Maps/map" + f + ".txt", true);
        writer.Write(sb.ToString());
        writer.Close();
        }
        Invoke("Scan", 0.2f);

    }

    void Scan()
    {
        pathfinder.GetComponent<AstarPath>().Scan();
        //foreach (GameObject agent in agents) agent.GetComponent<AgentController>().StartPath();
        //foreach (GameObject agent in agents) agent.GetComponent<Pathfinding.AIBase>().SearchPath();

        //player.GetComponent<Pathfinding.AIBase>().canMove = true;
    }

    // Update is called once per frame
    void Update()
    {

    }
}
