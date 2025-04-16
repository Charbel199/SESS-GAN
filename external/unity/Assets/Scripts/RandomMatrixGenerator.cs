using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.IO;

public class RandomMatrixGenerator : MonoBehaviour
{
    int f = 0;
    // Start is called before the first frame update
    void Start()
    {
        int rows = 10;
        int columns = 18;

        StringBuilder sb = new StringBuilder();
        for (int a = 0; a < rows; a++)
        {
            for (int b = 0; b < columns; b++)
            {
                sb.Append(Random.Range(0,3));
                sb.Append(' ');
            }
            sb.AppendLine();
        }
        while (System.IO.File.Exists("randomMap" + f + ".txt"))
        {
            f++;
        }
        StreamWriter writer = new StreamWriter("randomMap" + f + ".txt", true);
        writer.Write(sb.ToString());
        writer.Close();
    }

    // Update is called once per frame
    void Update()
    {

    }
}
